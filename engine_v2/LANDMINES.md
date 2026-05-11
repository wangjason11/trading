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

## Cycle-0 CTS Cannot Be Confirmed via Zone Proximity

**Rule:** The dual CTS proximity-confirmation path in
`MarketStructure._apply_pattern_at_apply_idx` (the per-candle check
guarded at the run loop in `_run_step`) is **disabled on cycle 0** by
the `st.cts_cycle_id > 0` gate. Cycle 0 must rely on pullback
confirmation. Don't remove this gate.

**Why this is a landmine:** `BOS_0` is `"initial_prior_extreme"` — the
swing extreme that existed before the structure began. Its distance
from `CTS_0` is uncontrolled and routinely small (12–25 pips observed
on NZD_USD H1). When `BOS_0`–`CTS_0` gap < ~2× `proximity_pips`, the
sd-inner proximity buffer overlaps the `CTS_0` price level itself. The
first candle after `CTS_ESTABLISHED` then trivially fires the trigger
even though price hasn't meaningfully retraced.

**Concrete cases that motivated the gate (NZD_USD H1, threshold 20p
at the time — since lowered to 9p in `DEFAULT_PROXIMITY_PIPS`):**
- sid=0 cyc=0: POI inner 0.56004 sat 12.2 pips below CTS=0.56126;
  threshold 20p → buffer extended 7.8p *above* CTS. Trigger fired at
  idx 116 (the next candle). Cascade: spurious cycle-1 transition,
  spurious `BOS_CONFIRMED @ idx 157`.
- sid=1 cyc=0: POI inner 0.58320 sat 12.5 pips above CTS=0.58195;
  buffer extended 7.5p below CTS. Trigger fired at idx 704. Cascade:
  spurious `BOS_CONFIRMED @ idx 705`.

**Why the gate stays even with the lowered 9p threshold:** the new
threshold reduces but does not eliminate the overlap pathology. A POI
inner within 9p of CTS still triggers the same failure mode; only the
SIZE of the geometric overlap shrinks. Cycle 0's BOS-CTS gap remains
unconstrained by definition, so the gate is the principled fix
regardless of threshold magnitude.

The same geometric overlap drives `subsequent_confluence` / `subsequent_counter`
over-firing in narrow-gap cycles (see `project_part4_blocker_135d.md` /
the Lever A draft). The MarketStructure gate handles cycle 0
specifically; cycles k>0 keep proximity confirmation enabled because
`BOS_k` is the pullback extreme from cycle k-1 and the gap is
structurally guaranteed.

**Why we don't apply a generic |inner − cts_price| > threshold gate:**
cycles k>0 don't need it (BOS-CTS gap is structurally meaningful) and a
generic gate would silently drop legitimate proximity confirmations on
borderline-but-valid POI placements. Cycle-0 is the only place the
guarantee is absent.

**If you must change the gate:** the spurious-cascade signature is two
events at the breakout idx — a `BOS_CONFIRMED` with `source: "pullback_extreme"`
and `pb_start: None` (meaning no pullback fired, the BOS came from the
proximity-only `[cts_confirmed_idx, breakout_apply_idx]` window). Grep
the structure_events.csv for that combination to catch regressions.

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

## Scenario 3 Pip Tolerance Scales with Timeframe

**Rule:** The BOS_0 probe exception pip tolerance must be appropriate for the timeframe:
- **H1:** 10 pips (default)
- **M15:** 3 pips
- **M5:** 1 pip (future)

**Why:** A 10-pip tolerance on M15 is too aggressive — it triggers false restarts because M15 price movements are smaller. The tolerance controls how close a candle must get to the BOS_0 zone outer bound to trigger a probe restart.

**Implementation:** Pass `pip_tolerance_pips` to `compute_structure_scenario_3()`. Lower-TF pipeline uses 3 for M15.

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

**Specific dependency (Mode C):** At a candle where both
`CTS_ESTABLISHED` (cycle n+1) AND `CTS_THRESHOLD_UPDATED` (cycle n) fire,
the alphabetical order processes `CTS_ESTABLISHED` FIRST (C_E < C_T). The
M15 reverse phase gate depends on this:
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

## Chart Entry Points: Use Registry, Not Positional df (Part 4 transitional)

**Rule:** `export_chart_plotly()` and `export_m15_chart_plotly()` accept
**both** a positional df fallback (`df=...` / `m15_df=...` / `h1_df=...`)
and a registry path (`registry=..., path_id=...`). **The positional
fallback is transitional** — kept so existing tests and ad-hoc inspection
scripts keep working. Migration plan §13.5.e removes it.

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
bypass those lookups, and once §13.5.e deletes the orchestrator's
deprecated `df.attrs` writes the positional path will silently produce
empty zone overlays.

**§13.5.c.iii update:** the `lower_tf_results` positional kwarg is
**removed** from `export_m15_chart_plotly`. The chart consumer reads
sub data from `m15_df.attrs["events" / "kl_zones" / "poi_zones" /
"fib_states" / "wave_candles" / "wvmi" / "prev_bos_lines"]` grouped by
`meta["entity_sid"]` per `m15_df.attrs["sids"]`. The same applies to
`export_chart_plotly`'s M15-zone overlay — it reads zones from the
M15.counter sub-entity in the registry, never from
`dfx.attrs["lower_tf_results"]`.

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
    source_kinds=["BOS"], fib_mode="m15_reverse",
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

## M15 Chart Sid-Tied Filter Uses `owner_by_idx` Per Rendered Candle

**Rule:** §13.5.c.iii implements spec §16.5's "most recent sid only per
candle" rule for sid-tied display elements (CTS dots, BOS markers, swing
lines, PB markers, prev_bos lines, wave-candle hover anchors) by checking
`owner_by_idx[rendered_candle_idx] == this_entity_sid`. The check uses
the **rendered candle's idx**, NOT the event's emission idx, because for
some events these differ:

| Element | Rendered candle |
|---|---|
| CTS_CONFIRMED dot | `meta["cts_anchor_idx"]` (the extreme), not `ev.idx` (the confirmation candle) |
| BOS_CONFIRMED dot | `ev.idx` |
| PB dot | `ev.idx` (pullback STATE_CHANGED event) |
| Wave-candle vertical line | `wc.last_wave_candle_idx` / `wc.first_wave_candle_idx` |
| Prev BOS line | `line_info["start_idx"]` |
| Swing-line extension at last sid | last idx still owned by this entity_sid (walk backward from `sid_rec.end_event_idx` skipping unowned cells) |

**Why this matters:** when var 4 cascades over var 2 starting at idx X,
var 2's CTS_CONFIRMED at idx 325 with `cts_anchor_idx=315` should still
render at 315 if 315 < X. Filtering by ev.idx=325 would hide it; filtering
by cts_anchor_idx=315 (the actual rendering candle) keeps it visible.

**`owner_by_idx` construction:** walk SidRecords sorted by entity_sid asc;
for each, claim `[creation_event_idx, end_event_idx]`. Later sids
overwrite earlier in the dict — `owner_by_idx[i]` always reflects the
most recent claimant. Built once per chart export by
`_compute_owner_by_idx`.

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

## Cascade Keys Off `entity_sid`, NOT `structure_id`

**Rule:** `_tag_old_sid_on_overwrite` in `multitf/entity_df_mutation.py`
matches prior-sid snapshots by `meta["entity_sid"] == prior_sid_id`.
NOT by `structure_id`. The two are different concepts:

| Concept | Meaning |
|---|---|
| `structure_id` (in event/zone meta) | The internal `MarketStructure` run id within one sub-build. Increments on internal reversals (sids 0, 1, 2, ... within one sub). Set by `MarketStructure.__init__(structure_id=0)`. |
| `entity_sid` (in event/zone meta) | The entity-wide monotonic sub id, assigned by the orchestrator at trigger time. Per spec §13.5.c sid numbering convention: sids are entity-wide, not per-parent-cycle. |

Multiple `entity_sid` values can share the same `structure_id` (every
sub-build starts MarketStructure with `structure_id=0`). Multiple
`structure_id` values can share the same `entity_sid` (a sub that
reverses internally produces sids 0 → 1 within MarketStructure but
both belong to the same entity_sid).

**Concrete trap:** "let me clean this up — `entity_sid` is just a
synonym for `structure_id`, right?" No. Cascading on `structure_id`
would mis-tag every sub's internal sid 0 as "the prior sid being
overwritten" the moment a new entity_sid lands.

**See also:** spec §13.5.c sid numbering convention block.
