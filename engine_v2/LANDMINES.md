# LANDMINES.md — Critical Constraints and Things to Avoid

> These are the rules that, if violated, will cause hard-to-debug issues. Read before making changes.

---

## What NOT To Do

- Don't rewrite modules without reading their spec first
- Don't add features outside current week's scope (park in `IDEA_PARKING_LOT.md`)
- Don't optimize before logic is visually validated
- Don't break existing event contracts
- Don't skip chart verification

---

## Pipeline Ordering Constraint

```
candle features → structure patterns → imbalance → market structure → KL zones → wave candles → Fib tracking → POI zones → WVMI → charting
```

**MUST:** Base features MUST run BEFORE market structure so zone resolution is stable.
**MUST:** WVMI MUST run AFTER POI zones — its gate (the first sd zone-proximity trigger from `check_zone_proximity`) depends on POI zone inner bounds.

**Why:** Market structure depends on candle classification and pattern detection from base features. The zone-proximity check examines both KL and POI zone inner bounds to find sd-direction triggers, so POI zones must exist first.

**Enforcement:** Pipeline ordering is defined in `pipeline/orchestrator.py` and marked as LOCKED.

---

## Event Contract Rules

Events are the communication backbone of the system. Breaking contracts causes cascading failures.

**Rules:**
1. **Never change event type names** — downstream consumers filter by exact string match
2. **Never remove fields from event.meta** — existing code may depend on them
3. **Adding fields is OK** — but document them in the relevant spec file
4. **Events are append-only** — never modify an event after it's emitted
5. **Events must include structure_id** — so downstream consumers can filter by structure

**Key events and their consumers:**
| Event Type | Primary Consumer |
|------------|------------------|
| STATE_CHANGED | Charting, Zones |
| CTS_CONFIRMED | Zones, Patterns |
| BOS_CONFIRMED | Zones, Patterns |
| RANGE_* | Charting (rectangles) |

---

## Structure ID Isolation Principle

**Rule:** When fixing issues in structure_id N+1, **never modify data for structure_id N**.

**What this means:**
- Don't change events that were emitted during sid N processing
- Don't overwrite df columns for rows that belong to sid N
- Don't alter zone boundaries established during sid N

**Why:** Each structure_id represents a complete, committed market regime. Retroactively changing sid 0 while debugging sid 1 leads to:
- Inconsistent event history
- Charts that don't match the underlying data
- Bugs that "fix themselves" when you run the full pipeline but reappear in isolation

**Safe approach:** If sid N has a bug, fix it in the code that processes sid N, then re-run the entire pipeline from scratch.

---

## DataFrame Column Overwrite Hazard

**Problem:** Columns like `market_state`, `structure_id`, `swing_dir`, etc. are overwritten when processing each structure. You cannot reliably query df columns for "which rows were in reversal for sid 0" after sid 1 has been processed.

**Landmine:** Code like this will fail silently:
```python
# WRONG: Returns empty or wrong results after sid 1 runs
rev_idx = df[df["market_state"] == "reversal"].index[0]
```

**Safe approach:** Use events for cross-structure queries:
```python
# RIGHT: Events preserve structure_id metadata
rev_event = next(e for e in events
                 if e.type == "STATE_CHANGED"
                 and e.meta.get("to") == "reversal"
                 and e.meta.get("structure_id") == target_sid)
rev_idx = rev_event.idx
```

---

## Zone Threshold Mutations

**Rule:** Zone boundaries should only change via THRESHOLD_UPDATED events, never by direct assignment.

**Why:** The charting system reads `bounds_steps` history to render zone expansions. Direct mutation skips this history, causing:
- Zones that appear at wrong sizes on chart
- Expansion timing that doesn't match actual price action

---

## Index Boundary Errors

Common sources of off-by-one bugs:

| Pattern | Risk |
|---------|------|
| `df.iloc[start:end]` | `end` is exclusive — double-check you're including the right candle |
| `range(start, end)` | Same — `end` is exclusive |
| `df.loc[start:end]` | `end` is **inclusive** — different from iloc! |
| Using confirm_idx vs start_idx | RANGE_STARTED has both — use start_idx for sort order, confirm_idx for timing |

---

## Dual CTS Proximity Confirmation — Cycle-0 + Rule 1 Are BOTH Required

**Rule:** Two independent gates guard the dual CTS proximity-confirmation
path in `MarketStructure._apply_pattern_at_apply_idx`. Both must pass for
sd-proximity to be eligible to confirm CTS:

1. **Cycle-0 carve-out:** `st.cts_cycle_id > 0` — proximity disabled on
   cycle 0 of every structure_id.
2. **Rule 1 (narrow-gap gate):** `|cts_price − bos_threshold| ≥
   min_gap_threshold` (per-TF `DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS`:
   H1=50, M15=30, M5=15 pips).

Don't remove either independently. The "both apply" decision (Q3 in the
2026-05-12 spec session) was deliberate: Rule 1 covers the geometric
pathology generically; the cycle-0 carve-out covers the additional
semantic concern that BOS_0 = `initial_prior_extreme` is a weaker
structural anchor than BOS_k (k>0) = prior cycle's pullback extreme,
even when its gap happens to be wide.

**The historical pathology that motivated cycle-0 carve-out:** `BOS_0`
is the swing extreme that existed before the structure began. Its
distance from `CTS_0` is uncontrolled and routinely small (12–25 pips
observed on NZD_USD H1). When `BOS_0`–`CTS_0` gap < ~2× `proximity_pips`,
the sd-inner proximity buffer overlaps the `CTS_0` price level itself.
The first candle after `CTS_ESTABLISHED` then trivially fires the
trigger even though price hasn't meaningfully retraced.

**Concrete cases (NZD_USD H1, threshold 20p at the time — since
lowered to 9p in `DEFAULT_PROXIMITY_PIPS`):**
- sid=0 cyc=0: POI inner 0.56004 sat 12.2 pips below CTS=0.56126;
  threshold 20p → buffer extended 7.8p *above* CTS. Trigger fired at
  idx 116 (the next candle). Cascade: spurious cycle-1 transition,
  spurious `BOS_CONFIRMED @ idx 157`.
- sid=1 cyc=0: POI inner 0.58320 sat 12.5 pips above CTS=0.58195;
  buffer extended 7.5p below CTS. Trigger fired at idx 704. Cascade:
  spurious `BOS_CONFIRMED @ idx 705`.

These cases are now blocked by BOTH gates simultaneously: the
geometric narrow-gap by Rule 1 (12.2p < 50p threshold), the structural
weakness by the cycle-0 carve-out.

**Why both, not just one:** Rule 1 alone leaves the rare "cycle 0 with
wide gap" case (gap ≥ 50p) where sd-proximity would be allowed despite
BOS_0's semantic suspectness. The cycle-0 carve-out alone leaves the
analogous narrow-gap pathology for cycles k>0 (where BOS_k-CTS_k can
also be narrow, just less commonly than cycle 0) unaddressed —
historically this surfaced as `subsequent_confluence` /
`subsequent_counter` over-firing in narrow-gap cycles (the parked
§13.5.d Lever A draft was attacking that same problem from a different
angle and is now largely subsumed by Rules 1-3).

**Retraction of prior reasoning:** an earlier version of this entry
argued "we don't apply a generic `|inner − cts_price| > threshold` gate"
because cycles k>0 supposedly didn't need it. That claim was wrong —
narrow-gap cycles k>0 had been silently mis-firing proximity
confirmations all along (cycle 0 was just the loudest case). Rule 1 is
exactly the generic gate that prior reasoning argued against. The
present design includes it.

**If you must change either gate:** the spurious-cascade signature is
two events at the breakout idx — a `BOS_CONFIRMED` with
`source: "pullback_extreme"` and `pb_start: None` (meaning no pullback
fired, the BOS came from the proximity-only
`[cts_confirmed_idx, breakout_apply_idx]` window). Grep the
structure_events.csv for that combination to catch regressions.

**Scenario 2 anchor agreement (closed 2026-05-13):**
MarketStructure's in-flight POI snapshot
(`zones/poi_zones.py::compute_poi_inners_for_cycle`) and FibTracker's
downstream POI derivation both pick Fib anchors via the shared pure
utility `zones/fib_tracker.py::select_fib_anchor_for_cycle`. The utility
embeds the Scenario 2 cross-cycle cond1/cond2/cond3 check, so for
`sid >= 1, cycle_id == 1` cases both layers agree on whether to anchor at
`(BOS_0, CTS_1)` (Scenario 2) or `(BOS_n, CTS_n)` (Scenario 3 / intra).
MS tracks the cycle-0 snapshot (`MarketStructureState.cycle0_data`)
populated at cycle-0 CTS_ESTABLISHED, refreshed on each CTS_UPDATED, and
locked at CTS_0 CONFIRMED — threaded into the resolver via the
`PoiInnersResolver` protocol's `c0_data` slot.

**Documented approximation:** MS does NOT track Scenario 1 (TRUE / revert
to FALSE on BOS_1 touching prev BOS zone outer). It passes
`scenario1=None` in c0_data, which routes the utility into the Scenario
2/3 evaluation path. FibTracker passes the post-revert `scenario1` value
through. When Scenario 1 is TRUE in FibTracker's view (rare — requires
CTS_0 idx >= reversal_confirmed_idx) AND Scenario 2 cond1/cond2/cond3 all
match (also rare), MS would pick cross-cycle anchors where FibTracker
picks intra. To plug that gap, thread the parent reversal_confirmed_idx
and prev BOS outer/sd through `structure_engine._make_market_structure`,
mirror the Scenario 1 resolution + revert logic on MarketStructureState,
and set `c0_data["scenario1"]` accordingly before each resolver call.

**Concrete signature of regression (keep for future debugging):** a
CTS_CONFIRMED dot whose `confirmation_method == "sd_zone_proximity"` BUT
the candle is visually nowhere near any rendered POI zone for that cycle
— suspect either the resolver has gone out of sync with FibTracker's
anchor selection, or `c0_data` isn't being threaded correctly through
`_refresh_poi_inners_for_cycle`. Trace by instrumenting
`_check_proximity_at_candle` to log the active `poi_inners_for_cycle` and
compare against `df.attrs["poi_zones"]`.

---

## Narrow-Cycle Rules 1+2+3 Are a Triple

**Rule:** The three narrow-cycle rules are interlocking. Removing any
one undoes the others' intent. They were designed together and must be
modified together if reconsidered.

| Rule | Where | What it does |
|---|---|---|
| 1 | `structure/market_structure.py::_apply_pattern_at_apply_idx` (the `cycle_gap_ok` guard) | sd-proximity cannot confirm CTS while gap < `min_gap_threshold` |
| 2 | `zones/zone_proximity.py::check_zone_proximity` (the `cycle_has_pullback_cts` check) | In narrow mode, scan triggers only fire on/after pullback-confirmed CTS |
| 3 | Same file, the `narrow_sd_fired` / `narrow_opp_fired` caps | In narrow mode, at most 1 sd + 1 opp_sd trigger per cycle |

**Why they're a triple:**

- Rule 1 → without it, sd-proximity confirms CTS spuriously in narrow
  cycles, kicking off premature cycle transitions (the original
  cycle-0 pathology generalized to all cycles).
- Rule 2 → without it, in mid-cycle-crossing scenarios (narrow then
  wide), narrow-mode triggers might fire BEFORE pullback CTS_confirmed
  exists — but Rule 1 guarantees no CTS_CONFIRMED yet, so the scan
  window doesn't even start. Rule 2 just makes the "no scan without
  pullback CTS in narrow mode" invariant explicit and robust against
  future changes to scan-window semantics.
- Rule 3 → without it, narrow cycles produce unbounded V/λ trigger
  cascades on borderline-overlapping geometry, defeating the purpose
  of restricting narrow-mode triggers in the first place.

**Monotonicity invariant they rely on:** the gap
`|cts_threshold − bos_threshold|` is monotonically non-decreasing
within a cycle (BOS extends in struct direction via probe; CTS extends
via range sync — both widen the gap). A cycle can transition narrow →
wide exactly once, never the reverse. So the post-facto scan can
evaluate the gap per-candle and apply Rule 3's caps only while narrow;
once crossed, alternation continues unrestricted (Q1 = A in the spec
session).

**Self-rescue prevention (Q7):** the post-facto scan evaluates gap at
start-of-candle (events with `idx < current_candle` applied), not at
end-of-candle, so an opp_sd-direction candle's own wick can't extend
`cts_threshold` enough via a same-idx `CTS_THRESHOLD_UPDATED` event to
lift itself out of narrow mode and qualify. The
in-MarketStructure Rule 1 check doesn't have this issue (sd-proximity
candles wick in the opposite direction and can't extend
`cts_threshold` on themselves), so it reads `st.cts.price` and
`st.bos_threshold` directly.

**Gap source MUST be events, NOT df columns:** the per-cycle gap is
reconstructed from `BOS_CONFIRMED` + `CTS_CONFIRMED` + their respective
`*_THRESHOLD_UPDATED` events filtered by `(structure_id, cycle_id)`.
DO NOT read `df["cts_threshold"]` / `df["bos_threshold"]` for this —
those columns are overwritten by the next structure when it begins
processing (see "DataFrame Column Overwrite Hazard" landmine). A
cycle's scan window can extend up to `reversal_apply_idx - 1`, which
crosses the boundary into the next structure's rows — by that point
the df cols are NaN or carry the next sid's state. Events are
append-only and tagged with their owning `(sid, cycle_id)`, so they
survive the boundary. Helpers:
`zones/zone_proximity.py::_build_cycle_threshold_timeline` +
`_apply_threshold_event`.

**The "no-pullback narrow cycle" outcome is intended, not a bug:** when
a cycle is narrow AND pullback never fires, the cycle gets no
proximity triggers AND no CTS_CONFIRMED event. WVMI silent. var3/var4
silent. Cycle proceeds to next BOS via breakout pattern only. This
mirrors today's cycle-0 silent behavior in narrow geometry.

**The "Lever A draft is now redundant":** the parked §13.5.d Lever A
(`MIN_LAMBDA_SPAN=5`) was attacking var3/var4 over-firing in
narrow-gap cycles by capping λ-span. Rule 3's "max 1 sd + 1 opp_sd"
directly caps the trigger count in narrow cycles, which is the same
effective restriction. Don't unpark Lever A without re-evaluating
against the post-Rules-1-2-3 baseline.

---

## WVMI Constraints

1. **Zero FB/FP volume blocks WVMI creation** — division by zero guard. Ensure candle features (volume) are computed before WVMI runs.
2. **Temp LP only locks on BOS_n+1** — do not assume `lp_locked=True` until BOS of the next cycle confirms. Until then, LP and pullback_momentum can shift every candle.
3. **buy_momentum/sell_momentum are direction-mapped** — for buy zones: buy=breakout, sell=pullback. For sell zones: reversed. Always check `zone_side` when interpreting.
4. **Zone proximity gate is mandatory (today)** — WVMI records are only created for cycles where the **first sd-direction zone-proximity trigger** fires (via `check_zone_proximity` in `zones/zone_proximity.py`). Scan window for that function: `[CTS_CONFIRMED confirmed_at, next_BOS_CONFIRMED confirmed_at - 1]` or `[CTS_CONFIRMED confirmed_at, REVERSAL apply_idx - 1]` (note: scan starts AT the CTS confirmation candle, not +1). Uses `ev.meta["confirmed_at"]` for both CTS and BOS (not `ev.idx` — see GOTCHAS "BOS_CONFIRMED ev.idx" entry). Uses only active POI zones at each candle (BOS KL zone is throughout-active). The orchestrator extracts the first sd trigger per cycle as the WVMI gate (`triggered_by_event_idx` in WVMIRecord.meta — Part 4 §8.7 attribution schema). Future refactor will rewire WVMI off this gate.

---

## Lower-TF Slice Must Include Lookback Buffer

**Rule:** When creating a DataFrame slice for lower-TF structures (e.g., M15), always include **at least 50 candles** before the structure start point.

**Why:** Multiple components depend on neighbor candles:
- `find_base_threshold` uses ±5 neighbor window → fails when `base_idx=0` (no left neighbors)
- Wave candle lookback needs 15-50 candles before the zone
- Imbalance fill checking needs prior candles

**Wrong:**
```python
trigger_df = m15_df.iloc[start_idx:end_idx + 1].copy()  # start_idx becomes 0
```

**Correct:**
```python
lookback = 50
slice_begin = max(0, start_idx - lookback)
trigger_df = m15_df.iloc[slice_begin:end_idx + 1].copy()
start_in_slice = start_idx - slice_begin  # Offset from slice start
```

**Symptom if wrong:** Zone bounds computed incorrectly (e.g., `find_base_threshold` returns None or wrong value), wave candles missing, or imbalance checks failing silently.

---

## Lower-TF Zones, POIs, and Fibs Must Be Capped at Lifecycle End

**Rule:** After running the downstream pipeline for a lower-TF structure,
cap every open-ended artifact to the M15 slice's last candle with
`deactivated_by: "lifecycle_end"`:
- **Zones:** any with `end_time=None`
- **POIs:** any with `end_time=None`
- **Fibs:** any with `active=True AND locked=False` (still-unlocked fibs
  that never saw CTS_n+1 CONFIRMED before the parent cycle ended, including
  pre-established cross fibs that never had a CTS_n+1 ESTABLISHED)

**Why:** The M15 structure's lifecycle is bounded by the parent H1 cycle
(ends at next BOS or reversal). Without capping, these artifacts render as
extending indefinitely on the chart — past the lifecycle boundary.

**Implementation:** In `lower_tf_pipeline.py`, after the downstream pipeline,
use `dataclasses.replace()` to cap each artifact kind:

```python
# Zones / POIs: cap by end_time
if zone.end_time is None:
    zone = replace(zone, end_time=last_time,
                   meta={**zone.meta, "active": False, "deactivated_by": "lifecycle_end"})

# Fibs: cap by active flag
if fib.active and not fib.locked:
    fib = replace(fib, active=False,
                  meta={**fib.meta, "deactivated_by": "lifecycle_end", "deactivated_at": last_idx})
```

---

## Probe `end_idx` Is the Supreme Bound

**Rule:** When `end_idx` is passed to a probe (`compute_structure_scenario_3`
Phase 1, Exception 2 loops in `compute_structure` / Scenario 3 Phase 2, or any
future similar probe), it is a HARD bound on the probe window. All inner
logic — termination conditions, exception-check windows, fallback paths —
must respect `end_idx` as the upper limit. Inner rules like "stop after 2
CTS_EST" can terminate the probe early but must NOT bypass or narrow logic
that operates within `end_idx`.

**Why:** `end_idx` is a caller-defined terminal that represents a known
real-world boundary (a WVMI activation candle, a reversal_confirmed idx,
a sub-structure lifecycle end). The probe's purpose is to validate the
start position GIVEN that bound. Bypassing the exception check because
"only 1 CTS_EST was found inside" silently skips validation in the very
window the caller cares about.

**Concrete case (Scenario 3 Phase 1, fixed 2026-05-13):** the exception
check window was hardcoded as `[cts_est[0].idx + 1, cts_est[1].idx]`. When
the probe window `[start_idx, end_idx]` contained only ONE `CTS_ESTABLISHED`
event, the code short-circuited Condition 4 to "finalized" without running
the exception check. For var1 sid=0 cycle=0 (probe `[96, 439]` on NZD_USD
H1), only one CTS fired (at idx 115); three candles in `[116, 439]` came
within the 3-pip tolerance of BOS_0 inner (idx 152, 157, 158) but were
never evaluated. Fix: when `end_idx` is defined, the exception check window
is `[cts_est[0].idx + 1, end_idx]` regardless of whether `cts_est[1]`
exists. Fallback to `cts_est[1].idx` only when `end_idx is None` (live
mode without an explicit terminal).

**Already-aligned locations:** Exception 2 in `compute_structure` (line
~231) and in `compute_structure_scenario_3` Phase 2 (line ~517) already
implement this principle — they use `reversal_confirmed_idx` (which IS
what's passed as `end_idx` to the inner MS probe) as the upper bound and
never narrow to a hypothetical `cts_est[1].idx`. Use those as the reference
pattern when adding new probes.

**If you add another probe with `end_idx`:** treat `end_idx` as the SOLE
upper bound for any candle-iteration inside the probe. Inner termination
events (reversal, 2 CTS_EST, etc.) can break the iteration early but must
not silently shrink the check window away from `end_idx`.

---

## Scenario 3 Pip Tolerance Scales with Timeframe

**Rule:** The BOS_0 probe exception pip tolerance must be appropriate for the timeframe. Canonical values live in `zones/zone_proximity.py::DEFAULT_PROBE_RESET_PIPS`:
- **H1:** 3 pips
- **M15:** 2.5 pips
- **M5:** 2 pips

**Why:** A wide tolerance on smaller timeframes triggers false restarts because lower-TF price movements are smaller. The tolerance controls how close a candle must get to the BOS_0 zone outer bound to trigger a probe restart. The invariant `DEFAULT_PROBE_RESET_PIPS[tf] < DEFAULT_PROXIMITY_PIPS[tf]` is asserted module-level in `zone_proximity.py` so probe-reset and proximity-trigger semantics never overlap.

**Implementation:** `compute_structure_scenario_3()` accepts an optional `pip_tolerance_pips: float` override; when None, it looks up the value from `DEFAULT_PROBE_RESET_PIPS` via the caller's `timeframe`. Type is `float` (not `int`) since the M15 default is fractional (2.5).

---

## Scenario 3 Constraints

1. **original_bos0_bounds captured at iteration 0 only** — subsequent probe iterations reuse the first BOS_0 zone for exception evaluation. Do not re-derive bounds mid-loop.
2. **Phase 2 only runs if status == "finalized" AND `run_continuation=True`** — when `run_continuation=False`, Phase 2 is skipped entirely (probe-only mode). The result contains only Phase 1 probe data.
3. **Exception evaluation checks inner bound, not outer** — proximity is measured as "candle high/low within tolerance of zone inner bound" (the bound closer to current price).
4. **Exception check window starts from CTS_EST + 1** — the CTS_ESTABLISHED candle itself is the pullback confirmation, naturally near the zone. Exclude it from the exception check (see GOTCHAS.md).
5. **Condition 4 split — `end_idx` is the discriminator:**
   - **4a) `end_idx is not None`** AND probe reached it without 2 CTS_EST → **finalized**. The caller-defined boundary is treated as a real terminal point (e.g., the first sd zone-proximity trigger candle is known and definitive).
   - **4b) `end_idx is None`** AND probe ran past available `df` data without 2 CTS_EST → **pending**. More candles may arrive later that resolve the probe; the caller can re-invoke with the same or advanced `start_idx`.
   - The probe never returns `pending` when an explicit `end_idx` was provided — the bound itself counts as a terminal break.
   - In current UC1 backtest, `end_idx` (the first sd zone-proximity trigger candle) is always set, so the pending path is dormant. Path becomes live for callers that pass `end_idx=None` (live mode where the future trigger candle isn't known yet).
   - Callers should treat `pending` as "skip downstream work for now" (e.g., `_run_h1_reverse_probe` returns `None` on pending so M15 isn't built).

---

## Lower-TF Pipeline Steps May Fail — Handle Gracefully

**Rule:** Both the H1 reverse probe (`compute_structure_scenario_3()`) and the M15 plain structure (`compute_structure_from_start()`) can raise `ValueError` or `IndexError`. The lower-TF pipeline must catch these at each step and return `None` (skip that trigger), not crash.

**Why it happens:** Some WVMI-activated cycles produce triggers where the structure reverses too quickly (e.g., cycle=0 in a short-lived structure). The `identify_start` function raises `ValueError` when it can't find CTS_CONFIRMED before a reversal.

**Implementation:** In `lower_tf_pipeline.py`, wrap both the H1 probe and M15 structure in try/except:
```python
# H1 reverse probe
try:
    s3_result = compute_structure_scenario_3(h1_df, ..., run_continuation=False)
except (ValueError, IndexError) as exc:
    return None

# M15 plain structure
try:
    m15_result = compute_structure_from_start(trigger_df, ...)
except (ValueError, IndexError) as exc:
    return None
```

---

## Event Sort Order Is a Dispatch Invariant

**Rule:** `sorted_events` in `_run_downstream_pipeline` is sorted by
`(e.idx, e.type)`. The alphabetical tie-break on event type is **relied upon
by handlers** — do not change the sort key without auditing downstream
dispatch logic.

**Specific dependency (`cross_cycle` mode, formerly Mode C / `m15_reverse`):**
At a candle where both `CTS_ESTABLISHED` (cycle n+1) AND
`CTS_THRESHOLD_UPDATED` (cycle n) fire, the alphabetical order processes
`CTS_ESTABLISHED` FIRST (C_E < C_T). The `cross_cycle` phase gate depends on this:
1. CTS_ESTABLISHED flips `_m15_phase[(sid, n+1)]` from `pre_established` to
   `established`
2. Subsequent CTS_THRESHOLD_UPDATED sees phase != `pre_established` and
   becomes a no-op — as intended (cycle n+1 is now in established mode)

If the sort were `(e.idx, -len(e.type))` or similar, `CTS_THRESHOLD_UPDATED`
would run first with phase still `pre_established`, triggering an extra
cross-fib check at an anchor that's about to be superseded. Silent
correctness drift, hard to notice in a chart.

**Rule of thumb:** When two events at the same idx have complementary
state-machine effects, ensure the state-flipping event sorts first. The
current alphabetical ordering happens to give us this for free; don't
regress it.

---

## MarketStructure Must Not Import From zones/

**Rule:** `engine_v2/structure/market_structure.py` has zero imports from
`engine_v2/zones/`. Zone-derivation primitives needed by the dual CTS
proximity check are consumed via the resolver protocol passed to
`MarketStructure.__init__`:

- `bos_inner_resolver: Callable[[pd.DataFrame, int, int], Optional[float]]`
- `poi_inners_resolver: Callable[[pd.DataFrame, int, float, int, float, int, int, int], List[float]]`

The orchestrator (`structure/structure_engine.py`) wires these resolvers
from the relocated derivation primitives in `zones/kl_zones_v1.py`
(`compute_bos_inner_from_event`) and `zones/poi_zones.py`
(`compute_poi_inners_for_cycle`). The `_make_market_structure` helper in
`structure_engine.py` is the only place the wiring lives.

**Why:** Zone derivation logically depends on structure events (zones are
DERIVED from events). A direct `structure/` → `zones/` import would invert
the dependency graph. Per Part 4 §13.5.b, that inversion was eliminated by
relocating the inline-derivation primitives to their natural home in
`zones/` and consuming them in MarketStructure via dependency injection.

**Implications:**
- Don't add imports from `zones/` inside `market_structure.py`. If new
  zone-derivation logic is needed, expose it as a callable resolver and
  wire it from `structure_engine.py`.
- `_check_sd_proximity_at_candle` (the pure proximity check) lives as a
  module-level private helper inside `market_structure.py` — it has no
  zones/ deps, just inner prices and a threshold.
- `structure_engine.py` is allowed to import from `zones/` — it's the
  orchestrator layer that sees both modules. The inversion ban applies
  only to `market_structure.py` (where the state-machine logic lives).

---

## Chart Entry Points: Use Registry, Not Positional df

**Rule:** both chart entry points are **registry-only** — the chart
positional-fallback portion of §13.5.e is DONE. `export_chart_plotly()` (H1)
and `export_m15_chart_plotly()` (M15) each require `registry=..., path_id=...`;
the legacy positional `df=...` / `m15_df=...` / `h1_df=...` fallbacks were
removed. (The *other* §13.5.e item — deleting the orchestrator's deprecated
`s_res.df.attrs[...]` writes — is NOT done: the H1 chart still reads its
overlays from `dfx.attrs[...]`, so those writes remain load-bearing.)

**Don't add new callers that pass positional df.** Use the registry path:

```python
export_chart_plotly(registry=registry, path_id="H1.main", title=..., ...)
export_m15_chart_plotly(
    registry=registry,
    path_id="H1.main >> M15.counter",  # M15 chart resolves its own
                                        # parent overlay via parent_of()
    title=..., ...,
)
```

**Why:** the M15 chart resolves its overlay via
`registry.parent_of(path_id)` per spec §16.3 ("each chart overlays only
its immediate parent, never grandparents"). The H1 chart's M15-zone
overlay reaches into `registry.get(f"{path_id} >> M15.counter").df.attrs
["kl_zones"]`. New callers that hand-pass `m15_df` + `h1_df` separately
bypass those lookups.

**§13.5.c.iii update:** the `lower_tf_results` positional kwarg is
**removed** from `export_m15_chart_plotly`. The chart consumer reads
sub data from `m15_df.attrs["events" / "kl_zones" / "poi_zones" /
"fib_states" / "wave_candles" / "wvmi" / "prev_bos_lines"]` grouped by
each snapshot's identity tuple `(parent_sid, parent_cycle_id, sub_sid)`
per `m15_df.attrs["sids"]`. The same applies to `export_chart_plotly`'s
M15-zone overlay — it reads zones from the M15.counter sub-entity in the
registry, never from `dfx.attrs["lower_tf_results"]`.

---

## Subordinate `mapping_sd` Must Use `-trigger.lower_sd`

**Rule:** In `multitf/lower_tf_pipeline.run_lower_tf_pipeline`, the
mapping step that picks the lower-TF candle within the validated parent
candle MUST use `mapping_sd = -trigger.lower_sd` (spec §4.3.1 unified
rule).

**Why this is a landmine:** for `first_counter`, `lower_sd = -parent_sd`,
so `-lower_sd == parent_sd` — meaning earlier code (`mapping_sd =
trigger.parent_sd`) produced the correct result. The two expressions are
mathematically equivalent for counter and the test suite passes either
way.

But for `first_confluence` (3b+), `lower_sd = +parent_sd`, so the two
expressions diverge:
- `-lower_sd = -parent_sd` ✓ correct (BOS extreme: lowest low in bullish
  parent, highest high in bearish)
- `parent_sd` ✗ wrong (would map to the parent CTS extreme, which is the
  wrong direction for a confluence sub)

A future "simplification" back to `trigger.parent_sd` silently corrupts
every confluence (and any future variation where `lower_sd != -parent_sd`)
without breaking any existing test. Always keep the unified `-lower_sd`
form.

**Generalization to deeper nesting:** the same principle holds for
descendants beyond depth 1. Always express the mapping in terms of the
sub being built, never in terms of the parent.

---

## Sub WVMI is Parent-Event-Driven; `source_kinds` is a Return-Only Filter

Two related load-bearing invariants in `pipeline/orchestrator._run_downstream_pipeline`,
both established in Part 4 Step 3d.iii. They look like "cleanups that shouldn't
change behavior" — they aren't.

**Rule 1 — `source_kinds` filters the returned list, NOT the derive call.**

```python
# RIGHT: derive everything, filter return
all_kl_zones = derive_kl_zones_v1(df, events, ..., source_kinds=None)
kl_zones = ([z for z in all_kl_zones if z.source_kind in source_kinds]
            if source_kinds else all_kl_zones)
# wave-candle loop iterates over `all_kl_zones`
```

```python
# WRONG (silent regression):
kl_zones = derive_kl_zones_v1(df, events, ..., source_kinds=source_kinds)
# wave-candle loop only sees BOS zones for subs → no CTS wave candles → 0 sub WVMI records
```

**Why:** `WVMITracker.on_cts_confirmed` requires BOTH a BOS wave candle and a
CTS wave candle for the same `(sid, cycle_id)`. CTS wave candles are derived
PER ZONE — if there are no CTS zones in the iteration set, `_find_wave_candle(...,
"CTS")` returns None and every sub WVMI sweep produces 0 records. This was a
latent bug from Week 8 Part 3 (sub WVMI silently empty) — fixed in 3d.iii.

**Rule 2 — Sub `_run_downstream_pipeline` calls pass `skip_wvmi=True`.**

```python
downstream = _run_downstream_pipeline(
    m15_result.df, m15_result.events, m15_result.struct_direction,
    source_kinds=["BOS"], fib_mode="cross_cycle",
    structure_path_id=sub_path_id,
    skip_wvmi=True,   # mandatory for subs
)
```

Sub WVMI is computed AFTER the `LowerTFResult` is built, by the orchestrator
calling `multitf/sub_wvmi.compute_parent_driven_sub_wvmi()`. The gate is parent
events per spec §8.3 / §8.4:

| Sub entity | Activated by | Spec |
|---|---|---|
| `H1.main >> M15.counter` (first_counter sids) | First var 3 trigger in same parent cycle | §8.4 |
| `H1.main >> M15.confluence` (var 1 sids) | Main first sd-prox in same parent cycle | §8.3 |
| `H1.main >> M15.confluence` (var 3 sids) | First var 4 trigger AFTER the var 3 in same parent cycle (else skip) | §8.3 |
| `H1.main >> M15.counter` (var 4-born sids) | First var 3 trigger AFTER the var 4 in same parent cycle (else skip) | §8.4 |
| Var 1 re-sweep on each var 4 fire | Deferred (option B) — batch is observably a no-op since records are deterministic given sub events; live mode will exercise this | §8.3 |

**Trap:** "Why does the sub also do its own zone proximity scan? Let me unify
those" — re-enabling entity-local sub WVMI inside `_run_downstream_pipeline`
would double-gate against the parent-event gate or silently revert to
entity-local gating. Don't.

**"Skip when no later gate exists" is intentional, not a bug.** Var 3 / var 4
sub sids that have no qualifying parent event AFTER them in the cycle get
empty `wvmi_records=[]`. Don't fall back to "the cycle's first var X" or
"some other sid's gate" — that breaks §8.7's one-trigger-per-record
attribution and silently invents activation events that didn't fire.
Empty here is the same outcome a proper §6.1 implementation would produce
in cycles where the gate genuinely doesn't exist.

---

## Var 3 + Var 4 Last-Per-Cycle Carve-Outs Are a Pair

> **⚠ Subsumed — sub-structure lifecycle redesign (2026-05-25).** The
> redesign's merge-and-bound build creates a sequential sid per
> `subsequent_*` trigger (bounded by the next), so these carve-outs vanish
> as a side effect — there's nothing to "remove" once it lands, and
> deactivation no longer goes through cascade/overwrite. Don't unpark the
> old §13.5.d removal as a standalone step. See
> `memory/project_sub_structure_lifecycle_redesign.md`.

**Rule:** Two last-per-cycle filters approximate spec §6.1's in-place
overwrite semantics until that mutation infrastructure lands. They MUST
be removed together.

| Filter | Where | Filters |
|---|---|---|
| `var3_last_per_cycle` | `_run_first_confluence_multi_tf` (orchestrator) | subsequent_confluence triggers |
| `var4_last_per_cycle` | `_run_multi_tf` (orchestrator) | subsequent_counter triggers |

**Why a pair:** spec §6.1 says each new var 3 sid overwrites the previous
open confluence sub sid; each new var 4 sid overwrites the previous open
counter sub sid. With overwrite-in-place not implemented, building every
var 3 / var 4 trigger would create N independent sids per parent cycle
(none deactivating any other) plus take ~15 min per replay because
many triggers are degenerate (input_idx == end_idx within 1-2 candles).
The carve-outs build only the LAST var 3 and LAST var 4 per
`(parent_sid, parent_cycle_id)` — at most one alive of each per parent
cycle, matching §6.1's effective "only most recent is alive" semantics.

**Symmetry:** removing one without the other leaves a half-finished
overwrite story. Either both stand or both fall.

**Detected-but-not-built triggers** are still exposed for inspection via:
- `df.attrs["subsequent_confluence_triggers"]`
- `df.attrs["subsequent_counter_triggers"]`

When §6.1 in-place overwrite lands: drop both filters, build all
triggers, let the overwrite path mark older sids `deactivated_by` as it
goes.

---

## Wrapping Logic in Loops: Preserve Post-Loop Behavior

**Rule:** When wrapping existing single-shot logic in an iteration loop, the behavior AFTER the loop must remain identical to the original code paths. The loop only changes what happens WITHIN iterations.

**Example:** Exception 2 probe was single-shot: exception → discard + outer loop; no exception → keep probe. When adding iterative re-probing, the post-loop paths must still map to the same two outcomes. Don't introduce new paths (like "keep the re-probe") that didn't exist before.

**Checklist when adding iteration to existing logic:**
1. Identify ALL exit paths in the original code (e.g., "exception triggered" vs "no exception")
2. Map each iteration outcome to the SAME original exit path
3. The iteration only refines WHICH value is used, not WHAT happens with it

---

## WVMI Records Carry Mixed-Coordinate Meta

**Rule:** `WVMIRecord` instances persisted on `entity_df.attrs["wvmi"]` mix
two coordinate systems and you MUST distinguish them when translating
indices:

| Field | Coordinate space |
|---|---|
| `fb_idx`, `lb_idx`, `fp_idx`, `lp_idx` (direct attrs) | Entity-df coords (sub-entity, e.g. M15) |
| `meta["triggered_by_event_idx"]` | **Parent-df coords** (e.g. H1 for an M15 sub) |
| `meta["lp_locked_at"]` (if present) | Entity-df coords |
| `bos_structure_id`, `bos_cycle_id` | Identity ints; coordinate-free |

**Why:** sub WVMI is parent-event-driven (spec §8.3 / §8.4 — see the
"Sub WVMI is Parent-Event-Driven" landmine above). The trigger event
(main first sd-prox, var 3, var 4) lives on the parent entity. The
record's wave-candle indices live on the sub entity. Both fields are
ints called "_idx", but they index different dataframes.

**Concrete trap (Part 4 §13.5.c.ii / §13.6):** any translation pass
that shifts WVMI indices when remapping a record between coordinate
spaces must skip `triggered_by_event_idx`. `mirror_lower_tf_result_to_entity_df`
documents this with an inline note; deeper-nesting future code that
re-maps records (e.g., when an M5 sub's parent is an M15 sub, not
H1.main) needs the same discipline.

**Rule of thumb:** before adding `meta["X_idx"]` to a record, decide
which df's index space `X` lives in and document it inline.

---

## Slice Copies Inherit Mirrored Structure Cols — Drop Before Passing to MS

> **⚠ Slated for removal — sub-structure lifecycle redesign (2026-05-25).**
> The slice + mirror + per-trigger `compute_imbalance` machinery this entry
> guards is replaced by the merge-and-bound build. Retained as current-code
> reference until the redesign lands. See
> `memory/project_sub_structure_lifecycle_redesign.md`.

**Rule:** In `multitf/entity_df_mutation.apply_trigger_to_entity_df`, the
slice-copy passed to `compute_structure_from_start` MUST drop every
column listed in `_STRUCTURE_COLS` AND `_MS_AUX_STRUCTURE_COLS` before
the call. The drop is REQUIRED, not optional.

**Why this is a landmine:**
`mirror_lower_tf_result_to_entity_df` (the c.i mirror, retained for
production in c.ii) initializes any new structure column on entity_df
with `entity_df[col] = pd.NA`, then writes actual values only inside the
prior sid's slice range. After the first mirror, `entity_df` has
`market_state` / `structure_id` / `range_hi` / etc. with `pd.NA` for all
rows outside that prior range.

For the next trigger's apply, `trigger_df = entity_df.iloc[a:b].copy() →
reset_index(drop=True) → compute_imbalance(...)` carries those NA
values into the slice. MS's `_ensure_output_cols` only initializes
columns that don't exist; it leaves the inherited NA values alone.

When MS's `_write_df_row` reaches its debug `range_break_frac`
calculation (line ~2057), it does:

```python
if self.df.at[prev_row, "range_hi"] is not None and ...:
    ...
    if c > float(self.df.at[prev_row, "range_hi"]):  # ← crashes
```

`pd.NA is not None` returns **True**, so the guard passes; then
`float(pd.NA)` raises `TypeError: float() argument must be a string or a
real number, not 'NAType'`.

**Mechanism applies to ALL structure cols, not just `range_hi`** —
`identify_start_scenario_2_after_reversal` reads `cts_event` /
`structure_id` / `cts_idx` from the same df; the outer loop's `rev_mask`
reads `market_state`. NA in any of these mis-routes scenario_2 or
silently picks up prior sids' rows.

**Fix (codified in `apply_trigger_to_entity_df`):**

```python
trigger_df = entity_df.iloc[slice_begin:m15_end_idx + 1].copy()
trigger_df = trigger_df.reset_index(drop=True)
trigger_df = compute_imbalance(trigger_df)
cols_to_drop = [
    c for c in (_STRUCTURE_COLS + _MS_AUX_STRUCTURE_COLS)
    if c in trigger_df.columns
]
if cols_to_drop:
    trigger_df = trigger_df.drop(columns=cols_to_drop, errors="ignore")
```

`_ensure_output_cols` then recreates the cols with proper int/float
defaults (`-1` for int idx cols, `float("nan")` for thresholds, `""` for
string event cols). MS reads prev_row consistently.

**Why mirror uses `pd.NA` in the first place:** `pandas` doesn't have a
single sentinel that's safe across object/float/int dtypes. `pd.NA` is
type-flexible. The proper fix is to revisit mirror's initialization to
match `_ensure_output_cols` defaults per col, but that's deferred — the
drop on the slice copy is the cheaper safeguard.

---

## MarketStructure Deep-Couples to Its Working DataFrame

> **⚠ Relevant but reframed — sub-structure lifecycle redesign (2026-05-25).**
> The redesign still must NOT pass a multi-sid entity df to MS — but its
> bounded single-structure run (per sid, stops at first reversal) changes
> how MS is invoked. The five `self.df` coupling points listed below remain
> the reasons MS can't take an entity df wholesale; the redesign addresses
> them via bounded per-sid runs, not the slice/mirror path. See
> `memory/project_sub_structure_lifecycle_redesign.md`.

**Rule:** Do NOT pass an entity df with prior sids' state directly to
`compute_structure_from_start`. MS owns its working df and assumes:

1. `_rewind_to(jump_to)` replays `from i = 0`, not from `start_idx`.
   On a slice with `reset_index`, idx 0 is the lookback boundary —
   harmless. On an entity df, idx 0 is the very first candle ever —
   MS would replay hundreds-to-thousands of unrelated candles, fire
   spurious patterns, and contaminate `self.df` cols.

2. `BreakoutPatterns(self.df)` precomputes / scans the full df. On a
   slice it sees only relevant candles. On an entity df it sees every
   candle since session start, including ones with no structural
   relationship to the new sid.

3. Outer loop's `rev_mask` (in `compute_structure_from_start`) is
   `(market_state=="reversal") & (structure_id==current_sid)` against
   the entire df. With prior sids' writes preserved (per §6.1), the
   mask matches prior sids' reversal rows — sending Scenario 2 to the
   wrong reversal_idx.

4. `identify_start_scenario_2_after_reversal` looks up CTS_CONFIRMED for
   prev_structure_id by scanning df cols across `df.index <
   reversal_idx`. With prior sids' data in those rows, it locks onto
   the wrong CTS or — worse — finds NA there and crashes (see the
   sibling landmine "Slice Copies Inherit Mirrored Structure Cols").

5. `_write_df_row` does `float(self.df.at[prev_row, "range_hi"])` on a
   debug-only path. Defaults are int (`-1`) on a fresh slice; on an
   entity df with mirrored cols, the value can be `pd.NA` (object
   dtype) and crashes.

**Implication:** the c.ii spec text described running
`compute_structure_from_start(entity_df, start_idx,
end_idx=lifecycle)` directly on the entity df. That direction is real,
but it requires an MS refactor to (a) rewind from `start_idx`, (b) bound
BreakoutPatterns to a window, (c) restrict scenario_2's mask to the new
sid's range, (d) handle pd.NA in debug paths. None of those landed in
c.ii; the slice + lookback + reset_index + per-trigger compute_imbalance
machinery stays. Slice-elimination is a deferred optimization, not
something achievable by changing the call site alone.

**Rule of thumb:** if you're tempted to "just pass the entity df to MS,"
list every place MS reads from `self.df` — there are at least five.

---

## `MarketStructure.run()` Crashes When `start_idx >= n` (latent)

**Rule:** Never call `MarketStructure.run()` (or any wrapper:
`compute_structure_from_start`, `compute_bounded_structure`,
`compute_structure_scenario_3`) with a `start_idx` at or past the end of
its working df. The `i >= n` early-return at `market_structure.py:316` is
broken.

**The bug:**

```python
def run(self):
    n = len(self.df)
    i = int(self.start_idx)
    if i < 0:
        i = 0
    if i >= n:
        return self.df, self.events, self.levels   # ← self.levels never assigned
```

`self.levels` does not exist — `levels` is only ever a *local* var
(`levels = self._events_to_structure_levels()`) on the normal path. So the
`start_idx >= n` branch raises `AttributeError: 'MarketStructure' object has
no attribute 'levels'` instead of returning an empty result.

**How it surfaces (runtime-confirmed 2026-05-25):** a short synthetic
fixture where `compute_structure_from_start` reversed early, then ran its
post-reversal Exception-2 probe from a start idx that landed past
end-of-data → crash inside the probe's `ms_probe.run()`. On the real
NZD_USD replay this is dormant because reversals occur mid-data with room
for the next structure.

**Forward-looking trap for Phase 2 (merge-and-bound sub build):** Phase 2
computes successor sid start idxs (next `subsequent_*` trigger / reversal
handoff). A handoff idx that lands at/after the entity df's last row will
hit this. Guard the start (`if start_idx >= n: return empty`) or fix
line 316 to return the local levels (`self._events_to_structure_levels()`
or `[]`) when stitching sids. `compute_bounded_structure` does not
re-derive starts, so it's only exposed if a caller passes an
out-of-bounds `start_idx`.

---

## M15 Chart Sid-Tied Filter Uses `owner_by_idx` Per Rendered Candle

**Rule:** §13.5.c.iii implements spec §16.5's "most recent sid only per
candle" rule for sid-tied display elements (CTS dots, BOS markers, swing
lines, PB markers, prev_bos lines, wave-candle hover anchors) by checking
`owner_by_idx[rendered_candle_idx] == this_sid_identity` (the identity
tuple `(parent_sid, parent_cycle_id, sub_sid)`). The check uses the
**rendered candle's idx**, NOT the event's emission idx, because for some
events these differ:

| Element | Rendered candle |
|---|---|
| CTS_CONFIRMED dot | `meta["cts_anchor_idx"]` (the extreme), not `ev.idx` (the confirmation candle) |
| BOS_CONFIRMED dot | `ev.idx` |
| PB dot | `ev.idx` (pullback STATE_CHANGED event) |
| Wave-candle vertical line | `wc.last_wave_candle_idx` / `wc.first_wave_candle_idx` |
| Prev BOS line | `line_info["start_idx"]` |
| Swing-line extension at last sid | last idx still owned by this sid (walk backward from `sid_rec.end_event_idx` skipping unowned cells) |

**Why this matters:** a sid's CTS_CONFIRMED at idx 325 with
`cts_anchor_idx=315` should render at the *anchor* 315, not the
confirmation candle 325. Filtering by ev.idx=325 would place/own the dot
at the wrong candle; using cts_anchor_idx=315 (the actual rendering
candle) is correct. (Pre-redesign this also mattered for cascade overlap;
merge-and-bound sids are now non-overlapping, but the rendered-vs-emission
idx distinction still stands.)

**`owner_by_idx` construction:** walk SidRecords sorted by identity tuple
`(parent_sid, parent_cycle_id, sub_sid)` asc; for each, claim
`[creation_event_idx, end_event_idx]`. Later sids overwrite earlier in the
dict — but since merge-and-bound sids are sequential & non-overlapping,
ranges don't actually overlap, so `owner_by_idx[i]` is just the single
owning sid. Built once per chart export by `_compute_owner_by_idx`.

**The default-keep heuristic:** `owner_by_idx.get(idx, eid) == eid` —
when a candle isn't covered by any SidRecord (e.g., outside every sub's
lifecycle), default to "owned by this sid" so the rendering doesn't
disappear. Practically rare today (most rendered idx fall inside some
sid's range), but the default is safer than skipping.

**Don't fall back to `ev.idx` filtering uniformly** — it loses the
cts_anchor_idx case. If you find yourself adding a new event-sourced
rendering, decide which idx to filter on: the emission idx or the
rendered idx.

---

## Mirror Translation of Nested-Dict Idx Fields Hardcodes Key Names

> **⚠ Slated for removal — sub-structure lifecycle redesign (2026-05-25).**
> The slice-local→entity-absolute mirror translation (and its hardcoded
> nested-dict idx-key handling) goes away with the merge-and-bound build,
> which builds directly in entity coords. Retained as current-code
> reference until the redesign lands. See
> `memory/project_sub_structure_lifecycle_redesign.md`.

**Rule:** `mirror_lower_tf_result_to_entity_df` in
`multitf/entity_df_mutation.py` shifts slice-local idx → entity-absolute
by adding `slice_begin`. Top-level meta keys are driven by the tuple
constants `_EVENT_META_IDX_KEYS` / `_ZONE_META_IDX_KEYS` (single source
of truth). **Nested-dict idx fields, however, are translated by
per-record special-case loops that hardcode the key name string** — and
each loop must match exactly one producer-side key.

Current nested-dict idx fields and their hardcoded loop keys:

| Producer site | Nested dict | Key in producer | Mirror loop key |
|---|---|---|---|
| `zones/kl_zones_v1.py` (INIT + expansion) | `meta["bounds_steps"][k]` | `"start_idx"` | `"start_idx"` |
| `zones/poi_zones.py` `_compute_poi_activation_history` | `meta["activation_history"][k]` | `"idx"` | `"idx"` |

**Hazard:** if the producer renames its key (or adds a new idx-bearing
key in a nested dict), and the mirror loop isn't updated, the special
case silently no-ops. Slice-local values get persisted as if they were
entity-absolute. The chart consumer reads them via `_lt_time(idx)` which
expects entity-absolute coords (`charting/export_m15_chart.py:594-598`),
and rectangles/dots land at completely wrong x-positions — but at
internally consistent ones, so no exception fires.

**Diagnostic signature:** sub chart element x-coords are offset from the
expected position by exactly `slice_begin` for the owning sub sid.
The rendered idx via `_lt_time` lands at `actual_idx - slice_begin`
(slice-local read as entity-absolute). The zone's hover *correctly*
shows the entity-absolute `base_idx` (because top-level meta IS shifted
by the tuple-driven path) — only the nested-dict positions are wrong.

**Concrete case (2026-05-18, fixed):** mirror loop iterated
`for step in new_meta["bounds_steps"]: if isinstance(new_step.get("idx"),
int): new_step["idx"] = ... + slice_begin`. Producer emits `"start_idx"`,
so the key check never matched. KL zone segments on M15.confluence
rendered with the leftmost expansion step at the correct base_idx (via
the `seg_x0 < x0` clamp at `export_m15_chart.py:845`), but subsequent
expansion-step rects landed at `entity_idx - slice_begin` instead of
`entity_idx`. For the sub `(0,0,0)` with `slice_begin=580`: zone base_idx=1761
with one expansion at entity 1898 rendered the second rect at M15 row
1318 (= 1898 − 580). Fixed by changing the loop's key check to
`"start_idx"`.

**Rule of thumb:** whenever adding or renaming a nested-dict idx field
in zones/POIs/fibs/etc., grep `entity_df_mutation.py` for the old key
name AND audit the per-record block of the touched type. The
top-level tuple constants do NOT cover nested dicts — those need their
own update.

---

## Fib + Scenario Imbalance Checks Are sd-Direction Strict

**Rule:** Every `has_unfilled_imbalance` call inside Fib activation,
Scenario 2 cond1/cond2/cond3, the cross-cycle dead-cycle walks, AND
MarketStructure's cycle-0 snapshot passes `direction=sd`. The only
permissive (no direction filter) call remaining in the codebase is
`get_unfilled_imbalances` in FibTracker's locking path (deferred —
doesn't affect behavior).

**Why this is a landmine:** the pre-2026-05-23 design was permissive,
with a documented rationale in POI_ZONES_SPEC §1 arguing that the
BOS→CTS span is structurally directional. A contributor reading old
commit messages, the old spec, or expecting "Fib activation is
permissive by convention" might "simplify" by removing the
`direction=sd` kwarg from these calls. That silently widens the input
set to include counter-direction imbalances which CANNOT produce POIs
(POIs are sd-direction by construction — POI_ZONES_SPEC §4).

**Consequence of accidental reversal:** Fib activation rate increases
slightly; Scenario 2 cond1/cond2/cond3 results may shift, causing
cross-cycle anchor selection to differ between MarketStructure's
in-flight resolver and FibTracker — re-introduces the Scenario 2
anchor agreement divergence closed 2026-05-13. Subtle; no test
failure unless someone has written a direction-mismatch test.

**Sites involved (15 total):**
- `zones/fib_tracker.py:120, 127, 275, 943, 1056, 1062, 1065, 1073,
  1132, 1137, 1139, 1158, 1620, 1650`
- `structure/market_structure.py:1791`

Plus `select_fib_anchor_for_cycle` takes `struct_direction` as a
parameter; the two callers (`compute_poi_inners_for_cycle` and
FibTracker's internal use) must pass it. Default value of 0 falls
back to permissive — kept for backward compat but no production caller
should hit it.

**See also:** `engine_v2/IMBALANCE_FILL_SEMANTICS.md` for the
canonical call-site matrix and `POI_ZONES_SPEC.md §1` for the
rationale.

---

## POI Activation Is Per-Candle, Not a Scalar Span

**Rule:** A POI's active interval is its `meta["activation_history"]` (a list
of `{"idx", "active", ...}` flips), NOT the scalar `meta["confirmed_idx"]`.
To decide "is this POI active at candle X?" call
`zones/poi_lifecycle.py::poi_active_as_of(zone, X)`. NEVER gate on
`confirmed_idx <= X <= end_idx`.

**Why:** `poi_zones.py` collapses the history to `confirmed_idx` = the LAST
activate idx (overwritten per activation, never cleared on deactivation). A
POI commonly flaps active→inactive→active within one cycle, so the scalar
points at the final stretch and hides every earlier one. A gate that uses
`[confirmed_idx, end_idx]` as one interval silently drops triggers that
should fire in an earlier active stretch.

**What broke (2026-05-26):** the `sd:POI` proximity gate used the scalar and
lost the sid1-cyc2 candles at idx 926 (`sd`) and 954 (`opp_sd`) — the POI was
active at 926 (stretch `[905,951]`) but `confirmed_idx` had collapsed to 997.
Full causal chain in GOTCHAS "POI `confirmed_idx` Is a Lossy Scalar".

**The single source of the per-candle walk** is `zones/poi_lifecycle.py`
(`active_stretches_from_history`, `poi_active_as_of`,
`poi_confirmed_idx_as_of`). It is a pure leaf module — imports nothing from
`zones/` or `charting/`, so both layers can depend on it without a cycle.
`charting/_zone_render.compute_poi_active_stretches` already delegates to it;
keep new consumers on the same helper so chart fills, the proximity gate, and
hover labels never disagree. `confirmed_idx` remains ONLY as a legacy chart
fallback (zones predating `activation_history`) and the debug print — do not
revive it as an activation bound, and do not "fix" the bug by flipping the
producer to first-activate (that ignores the deactivation flaps and
over-activates).
