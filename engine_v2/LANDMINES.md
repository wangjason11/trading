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
**MUST:** WVMI MUST run AFTER POI zones — the main's gate (the first sd zone-proximity trigger from `check_zone_proximity`) depends on POI zone inner bounds (a sub's WVMI is ungated since Plan G, but it runs in the same slot: it needs the KL zones and wave candles).

**Why:** Market structure depends on candle classification and pattern detection from base features. The zone-proximity check examines both KL and POI zone inner bounds to find sd-direction triggers, so POI zones must exist first.

**Enforcement:** Pipeline ordering is defined in `pipeline/orchestrator.py` and marked as LOCKED.

---

## Event Contract Rules

Events are the communication backbone of the system. Breaking contracts causes cascading failures.

**Rules:**
1. **Never change event type names** — downstream consumers filter by exact string match.
2. **Never remove a field from `event.meta`.** Existing code may depend on it.
3. **Never rename an `event.meta` key, or change the meaning of an event field (`ev.idx`, `ev.price` or a meta
   key), except by an atomic migration.**
   - The migration is declared in its commit message together with its `/compare` prediction: the exact CSV cells
     and run.log lines that may change.
   - The **migration commit** — the one in which the emitted key or meaning changes — also moves every site that
     reads **or writes** the field (production, debug, charts, test fixtures), every registry that lists it (e.g.
     `_EVENT_META_IDX_KEYS`), and every current doc and skill that names it. Dated records (landed plans, save
     folders, commit messages) are history and stay as written; memory is updated at the same checkpoint.
   - The emitter sets the key on every event of that type (`None` allowed). Every `.get(key)`,
     `.get(key, default)` and `key in meta` read of it becomes `meta[key]`. **No alias** is kept.
   - Readers may be made independent of a meaning change in earlier commits (e.g. by routing them through an
     accessor). The migration commit must then carry a proof that no reader still depends on the old meaning: a
     variant replay or test whose diff is exactly the declared cells.
4. **Adding fields is OK** — document them in the relevant spec file.
5. **Events are append-only** — never modify an event after it's emitted
6. **Events must include structure_id** — so downstream consumers can filter by structure

Rule 3 was approved by the user on 2026-09-24 (Plan E §4.2). First use: E1, the pattern-realm event key
`anchor_idx` → `pattern_anchor_idx` on `CTS_ESTABLISHED` / `REVERSAL_CANDIDATE` / `REVERSAL_WATCH_START`.
`tests/test_event_meta_idx_keys.py` guards the registry half: an index-valued event / zone / fib / wave-candle meta
key must be in `_EVENT_META_IDX_KEYS` / `_ZONE_META_IDX_KEYS` / `_FIB_META_IDX_KEYS` / `_WAVE_CANDLE_META_IDX_KEYS` —
no slice-local allow-list since Post-E·2 (2026-09-26, PLAN_E §9.3). It also scans every event-meta key
`market_structure` writes (Plan E E2a) and every `meta=` key the KL / POI / fib / wave-candle emitters write
(Post-E·2), so a key the fixture never produces is covered, and it pins the VALUES (each mirrored element == its
slice-local source + `slice_begin`). Since the Post-E·2 landing review it also checks the lists themselves (index-like
names only, never a known non-index, every entry emitted by its element kind), every mirror shift site on a synthetic
result, a broad emitter scan (`**{...}` splats, attribute subscripts, `setdefault`) and every int meta value by type.
Rule 4 use, Plan E E2a (2026-09-24): `CTS_ESTABLISHED.meta["cts_anchor_idx"]` and
`BOS_CONFIRMED.meta["bos_anchor_idx"]` (documented in ARCHITECTURE "`ev.idx` convention"). Rule 3 meaning
change, Plan E E4a (2026-09-25): `CTS_ESTABLISHED.idx` := the MOMENT (`== meta["confirmed_at"]`, asserted at
the emit; `ev.price` stays the anchor's price); proof = the E4 variant replays (`FLIP=est` == the landed
`/compare`: 3 events `idx` cells) + `tests/test_e4_simulation.py`; Plan E E4b (2026-09-25): `BOS_CONFIRMED.idx`
:= the MOMENT likewise (`FLIP=bos` == the landed `/compare`: 34 events `idx` cells). **Tests:** every
`CTS_ESTABLISHED` / `BOS_CONFIRMED` built during a test must satisfy the contract — `tests/conftest.py` validates
each construction (keys present as ints, `idx` == the index `EVENT_IDX_IS` names for the type: the moment on
both since Plan E E4a / E4b); build them with
`tests/_event_factory.py` (its `idx` default follows the same table), or mark a deliberately illegal test
`@pytest.mark.illegal_event_contract`. A `mutate=` hook that edits `confirmed_at` after construction bypasses
the validator: it must move the event's `idx` with it and re-run `validate_event_contract` on the
edited events (Plan E E4a review pins P1/P2 — a dropped `idx` move had left an illegal event nobody noticed).
Rule 3 meaning change on a non-event record, Plan G (2026-09-30): a SUB `WVMIRecord`'s `meta["triggered_by_event_idx"]` /
`["triggered_by_event_type"]` := the lens's first WVMI-class parent trigger inside the sub's window — attribution,
stamped per lens after the mirror, None when none lands there (before: the one trigger whose sweep created the records,
never None); main rows keep the gate meaning. Proof = the keyed WVMI `/compare` == PLAN_G §5 (the counter CSV's sub 3
710 → 871 and sub 7 1020 → None; `review_scripts/cmp_wvmi_keyed.py`).

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
`source: "pullback_extreme"` and `last_pullback_apply_idx: None` (`pb_start` before Post-E·4; meaning no pullback
fired, the BOS came from the proximity-only
`[cts_confirmed_idx, breakout_apply_idx]` window). Grep the
structure_events.csv for that combination to catch regressions.

**Scenario 2 anchor agreement (closed 2026-05-13):**
MarketStructure's in-flight POI snapshot
(`zones/poi_zones.py::compute_poi_inners_for_cycle`) and FibTracker's
downstream POI derivation both pick Fib anchors via the shared pure
utility `zones/fib_tracker.py::select_fib_anchor_for_cycle`. The Scenario 2
cross-cycle cond1/cond2/cond3 check is, since §11a (2026-06-17), DELEGATED to
the shared pure routine `zones/cross_cycle_fib.py::resolve_cross_cycle_eligibility`
(single-step `target=1`, `fill_as_of="snapshot"`). Fill horizons are in lock-step
too: cond1's is the triggering event's MOMENT in both layers (FibTracker: the
handled event's; MS: `_refresh_poi_inners_for_cycle(moment_idx)` — Plan E E3a),
cond2 (the cycle-0 caches: the moment of their last write) and cond3
(`snapshot_horizon_idx` = the BOS_1 moment == the CTS_1 ESTABLISHED moment) since
Plan E E3a′ — any change moves both layers in the same change.
`select_fib_anchor_for_cycle`
is now a thin wrapper applying only the main-only Scenario-1 outer gate around
it. So for `sid >= 1, cycle_id == 1` cases both layers agree on whether to
anchor at `(BOS_0, CTS_1)` (Scenario 2) or `(BOS_n, CTS_n)` (Scenario 3 / intra)
— except the knowability divergence (M1) below.
MS tracks the cycle-0 snapshot (`MarketStructureState.cycle0_data`)
populated at cycle-0 CTS_ESTABLISHED, refreshed on each CTS_UPDATED, and
locked at CTS_0 CONFIRMED — threaded into the resolver via the
`PoiInnersResolver` protocol's `c0_data` slot. Its `has_unfilled` and
FibTracker's cycle-0 cache (`_cross_cycle_data[sid]["cycle0"]["has_unfilled"]`)
are both Scenario-2 **cond2** and both stored **UNCUT** (`evaluated_at=None`,
Plan F): their only reader is `select_fib_anchor_for_cycle`
(`prior_cached_liveness`) at cycle 1 only (sid ≥ 1) — FibTracker at CTS_1 ESTABLISHED, MS
at each cycle-1 refresh — i.e. after CTS_0, when every gap in `[BOS_0, CTS_0]`
has formed. Keep BOTH uncut or cond2 diverges. A FibTracker decision taken
at a write event itself — the Scenario-1 cycle-0 activation — uses the value
cut at that event's moment (at CTS_0 ESTABLISHED `_on_cts_established`'s own
`has_unfilled`; on update `FibTracker._c0_has_unfilled_now`), never the cache.

**Documented exception — the knowability divergence (M1, accepted
2026-09-24, Plan F):** FibTracker asks every decision read at the handled
event's moment (`evaluated_at` = `event_fields.event_moment(ev)`; its two
cycle-0 cache writes stay uncut); the MS
in-flight resolver passes `evaluated_at=None` (its snapshot is read only at
`i > st.cts.idx` — MARKET_STRUCTURE_SPEC "Snapshot vs per-candle"). cond2
(cached, uncut in both) and cond3 (its window ends at CTS_0, before any moment)
still agree; **cond1 and the fib's existence can diverge**. Drop case: the ONLY
sd gap of a lag-0 CTS_ESTABLISHED or a raw CTS_UPDATED has c2 == the event
candle → FibTracker creates no fib at that event, while MS — which builds its
fib as always-active in `compute_poi_inners_for_cycle` — keeps a POI inner,
usable for sd-proximity CTS confirmation from the next candle. Permanent on H1
cycle ≥ 1 (fib activation is one-shot at EST); on subs until a later
CTS_UPDATED re-asks. **0 cases on the reference window.** The same class
already existed: a gap that forms after an H1 cycle-≥1 EST enters the MS inners
at the next CTS refresh but never activates the downstream fib (not measured).
Pinned by `tests/test_imbalance_c3_knowability.py::test_m1_ms_inflight_keeps_the_inner_fibtracker_creates_no_fib`
(flips when the "re-ask when the gap's c3 closes" follow-up lands, PLAN_F §7).
Canonical: IMBALANCE_FILL_SEMANTICS.md "Knowability — the c3 rule".

**Documented approximation:** MS does NOT track Scenario 1 (TRUE / revert
to FALSE on BOS_1 touching prev BOS zone outer). It passes
`scenario1=None` in c0_data, which routes the utility into the Scenario
2/3 evaluation path. FibTracker passes the post-revert `scenario1` value
through. When Scenario 1 is TRUE in FibTracker's view (rare — requires
the CTS_0 event's moment >= reversal_confirmed_idx) AND Scenario 2 cond1/cond2/cond3 all
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

## `_activate_fib`'s Versioned-Cross Obsolete Is H1-Only

**Rule:** The §11a-ii addition in `FibTracker._activate_fib` that obsoletes a
**prev-cycle versioned cross** (so the next cycle's activation sets the
`new_cycle` terminal on the now-versioned H1-main cross) is gated to
`self.fib_mode == "h1"` and MUST stay gated.

**Why:** `_activate_fib` is shared by both modes. Subordinate (`cross_cycle`)
crosses are obsoleted by `_m15_create_cross` → `_obsolete_prev_cycle_all_fibs`
at the *next cross's creation*. A sub `cross_failed` single also reaches
`_activate_fib`; if the versioned-cross obsolete ran there too, it would stamp
`obsolete_reason`/`_set_terminal(...,"new_cycle")` on the prior sub cross,
flipping its `end_reason` from `next_cycle` (the finalize pass-through) to
`new_cycle`. Caught by `/compare` as a 1-row diff in
`M15_{counter,confluence}_fib_lifecycle.csv` (`end_reason: next_cycle→new_cycle`).
Only the H1-main cross needs this hook (subs handle their own), so the `"h1"`
gate is both necessary and sufficient.

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
cycle's scan window can extend up to `reversal_idx - 1`, which
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
2. **Temp LP only locks on BOS_n+1** — do not assume `lp_locked=True` until BOS of the next cycle confirms. Until then, LP and pullback_momentum can shift every candle — within `[FP+1, cycle end − 1]` on main AND on every sub projection since Plan G (the data end while the cycle is open; `WVMI_SPEC` "Temporary LP Selection"). A lock LP the tracker's frame cannot read (BOS_n+1's last wave candle past a capped sub's end, or past the data edge) is never taken: the record locks with its temp LP (Plan G Q10).
3. **buy_momentum/sell_momentum are direction-mapped** — for buy zones: buy=breakout, sell=pullback. For sell zones: reversed. Always check `zone_side` when interpreting.
4. **Zone proximity gate — the MAIN only** — main WVMI records are only created for cycles where the **first sd-direction zone-proximity trigger** fires (via `check_zone_proximity` in `zones/zone_proximity.py`). Scan window for that function: `[CTS_CONFIRMED confirmed_at, next_BOS_CONFIRMED confirmed_at - 1]` or `[CTS_CONFIRMED confirmed_at, reversal_idx - 1]` (note: scan starts AT the CTS confirmation candle, not +1; the reversal term is the sid's REALISED `STATE_CHANGED(to=reversal)` candle — `compute_reversal_idx_by_sid`, since 2026-09-28 — never a `REVERSAL_CANDIDATE`'s scheduled `apply_idx`, a prediction a watch expiry can discard; GOTCHAS "Key boundaries"). Uses `ev.meta["confirmed_at"]` for both CTS and BOS (not `ev.idx` — see GOTCHAS "BOS_CONFIRMED ev.idx" entry). Uses only active POI zones at each candle (BOS KL zone is throughout-active). The orchestrator extracts the first sd trigger per cycle as the WVMI gate (`_first_sd_prox_gate`; `triggered_by_event_idx` in WVMIRecord.meta — Part 4 §8.7 attribution schema). **Subs are ungated since Plan G (2026-09-30):** every CTS_CONFIRMED of a rendered sub gets a record (`_run_downstream_pipeline(wvmi="none")`), and zone proximity is never run for a sub — see "Sub WVMI Is Computed Inside the Projection" below.

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

**Rule (Plan C, 2026-09-20 — restated on the pool model; PART4 §17.9):** every
KL zone / POI / fib / wave-candle element of a sub is derived over the unique
sub's real-time lifecycle window and MUST end at the sub's `end_idx`. The cap
is not a post-hoc pass: `entity_df_mutation.render_sub_projection` passes the
sub's `end_idx` (slice-local, `sub.end_idx - slice_begin`) as `lifecycle_cap`
and `sub.end_reason` as `cap_reason` into `pooled_structure_build.project_to_window`
→ `orchestrator._run_downstream_pipeline` → `structure_lifecycle.compute_cycle_lifecycle`,
whose per-`(sid, cycle)` end is `min(next-cycle clamped start, reversal,
lifecycle_cap)`. KL / POI / fib records INHERIT that `end_idx` / `end_reason`
(pass-through — see "Sub Lifecycle-Start Clamp … Start AND End Resolution Are
Shared" below); the KL derivation drops any legacy `active` / `deactivated_by`
keys from zone meta (`kl_zones_v1`, the lifecycle block), and no derivation
writes `deactivated_by` for a LIFECYCLE end (`fib_tracker` still writes it for
its version-internal `cross_failed` / `cross_shortened` / `scenario1_revert`
supersedes — a different mechanism, see "FibState Lifecycle Gate Is
Per-Record"). An open sub (`end_idx is None`) passes `lifecycle_cap=None`
and its elements stay open to the data edge, exactly like main.

**Vocabulary:** a sub's `end_reason` is one of `{reversal,
same_dir_replacement, parent_end}` (or `None` while open) — the reason of the
record end that won the §17.5 aggregation. That is the string the zone / POI /
fib `end_reason` carries for an element the cap ended. `deactivated_by` and
`"lifecycle_end"` are gone from every artifact (`"lifecycle_end"` survives only
as the dead default of the `cap_reason` parameter — never reached, because a
sub with a cap always passes its own `end_reason` and main passes no cap).
`next_cycle` (an end produced by the next sub cycle on the same sub, not by
the cap) is unchanged and stays internal to the zone layer.

**Why (stronger under the pool):** sub geometry now runs to the DATA EDGE
(`build_or_get_geometry`, run cap = `len(m15) - 1`), not to the parent
boundary. Without the cap every element of a sub that ended (by its own
reversal, by a same-lens replacement, or because its parent cycle ended)
would render to the far-right edge, across candles the sub was no longer
tradeable on. The cap is the only thing bounding derived elements to the
lifecycle; the geometry no longer does it for free.

**History (pre-Plan C; `multitf/lower_tf_pipeline.py`, deleted by Plan C):**
the original rule was a post-hoc `dataclasses.replace()` loop after the
downstream pipeline — zones / POIs with `end_time=None` and fibs with
`active=True AND locked=False` were closed at the M15 slice's last candle with
`meta["deactivated_by"] = "lifecycle_end"`. The slice was itself bounded by the
parent cycle's end, so the slice edge and the lifecycle end coincided. B2
(2026-05-27) moved the cap into the derivations as `lifecycle_cap`; Plan C
decoupled it from the run bound (the run bound is now the data edge, the cap
is the sub's aggregated `end_idx`).

---

## Probe `end_idx` Is the Supreme Bound

> **Rename note (Plan C, 2026-09-20):** the probe's search bound is now spelled
> **`probe_end_idx`** — the parameter of `unified_probe` / `_run_phase1` /
> `_run_phase2` (`structure/unified_probe.py`), `_probe_with_cache`'s parameter,
> `ProbeCacheEntry.probe_end_idx` and the `[probe_cache]` log strings (the H1 value FC price-maps into
> it — `FirstConfluenceTrigger` field / `MultiTFTrigger.meta` key — carried the
> same name until Plan E Post-E·3, 2026-09-27: now `parent_cts_anchor_idx`). It is a COMPUTE bound
> (the inclusive upper edge of the search window, like the run cap) and has
> nothing to do with the lifecycle `end_idx` of a `TriggerRecord` /
> `PooledStructure` (PART4 §17.4–§17.5) — that name collision is why it was
> renamed. The MS bound keeps its name: `MarketStructure(end_idx=...)` /
> `compute_bounded_structure(end_idx=...)`. The probe's OUTPUT anchor is
> `ProbeResult.starting_idx` (was `start_idx`) — a HISTORICAL field, the pool
> key, never a lifecycle value. Everything below still holds; read `end_idx`
> in this entry as the probe bound (`probe_end_idx`) unless it names the MS
> parameter.

**Rule:** When `end_idx` is passed to a probe (`compute_structure_scenario_3`
Phase 1, Exception 2 loops in `compute_structure` / Scenario 3 Phase 2, or any
future similar probe), it is a HARD bound on the probe window. All inner
logic — termination conditions, exception-check windows, fallback paths —
must respect `end_idx` as the upper limit. Inner rules like "stop after 2
CTS_EST" can terminate the probe early but must NOT bypass or narrow logic
that operates within `end_idx`. Plan B's early stop at the 2nd CTS_EST
(`MarketStructure(stop_after_cts_established=2)` in `unified_probe` Phase 2,
landed 2026-09-20) is the sanctioned form of "stop after 2 CTS_EST": it ends
the run but the retrace window it classifies lies before the 2nd CTS, so
nothing inside `end_idx` is narrowed — and it is *in addition to* the bound
(`n_cts <= 1` runs still reach `end_idx`), never instead of it.

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
mode without an explicit terminal). (`.idx` was the CTS anchor then; the code
reads `ef.cts_anchor_idx(...)` since Plan E E2b — an EST's `.idx` is its moment
since E4a.)

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

**The MS side of the same rule (Plan A, 2026-09-19):** the MarketStructure run
a probe drives is itself bounded with truncation semantics — see "Bounded MS
Runs Must Not Read Past `end_idx`" — and `unified_probe` Phase 1's detector is
bounded at `end_idx` too. **Known exception (L5b, deliberately open):** the
probe's ad-hoc BOS_0 zone derivation at a reset candidate
(`unified_probe._bos0_inner_at_start` → `build_ad_hoc_bos0_reference_zone` →
`identify_base_pattern`) still reads up to 5 candles past `end_idx`. It is a
reproducibility gap against the stated bound (not live causality — those
candles exist at trigger time); closing it can move the H1 sid-1 start via the
main reversal probe, so it needs its own decision and `/compare`.

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
4. **Exception check window starts from the CTS anchor + 1** (`ef.cts_anchor_idx(cts_est[0]) + 1`; the two Exception 2 loops do the same — test-only paths that keep the anchor, PLAN_E Q10). That anchor (`meta["cts_anchor_idx"]`; `ev.idx` until Plan E E4a) is the cycle-0 CTS **anchor** — the pattern extreme, the first argmax(`h`) / argmin(`l`) over the breakout pattern's span — NOT a pullback confirmation and NOT the moment the cycle was established (`meta["confirmed_at"]` = `ev.idx` since E4a, the apply candle; ARCHITECTURE "`ev.idx` convention"). The anchor candle belongs to the breakout itself, and the check asks whether price returns to the zone *after* the breakout, so exclude it (see GOTCHAS.md). When the CTS anchor precedes the apply candle, the window's first candles (up to `confirmed_at`) are still inside the breakout pattern.
5. **Condition 4 split — `end_idx` is the discriminator:**
   - **4a) `end_idx is not None`** AND probe reached it without 2 CTS_EST → **finalized**. The caller-defined boundary is treated as a real terminal point (e.g., the first sd zone-proximity trigger candle is known and definitive).
   - **4b) `end_idx is None`** AND probe ran past available `df` data without 2 CTS_EST → **pending**. More candles may arrive later that resolve the probe; the caller can re-invoke with the same or advanced `start_idx`.
   - The probe never returns `pending` when an explicit `end_idx` was provided — the bound itself counts as a terminal break.
   - No production code calls `compute_structure_scenario_3` any more (only tests); when UC1 did, `end_idx` (the first sd zone-proximity trigger candle) was always set, so the pending path was dormant. Path becomes live for callers that pass `end_idx=None` (live mode where the future trigger candle isn't known yet).
   - Callers should treat `pending` as "skip downstream work for now" (e.g., `_run_h1_reverse_probe` — since removed; `compute_structure_scenario_3` has only test callers now — returned `None` on pending so M15 wasn't built).

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

**Where the rule lives now (Plan C, 2026-09-20 — `lower_tf_pipeline.py` is
deleted):** the same two seams survive as the sweep's two failure reasons
(PART4 §17.7). The probe side: every resolver (`_resolve_trigger_m15_start`,
`_resolve_reversal_start` in `multitf/entity_df_mutation.py`) returns a
`ProbeFailure` instead of raising → the trigger is logged as
`UnresolvedTrigger(reason="probe_failed")`. The MS side: `_build_geometry`
wraps `compute_bounded_structure` in `try/except (ValueError, IndexError)` and
returns `None` → `build_or_get_geometry` creates NO pool entry (no `sub_id`
consumed) and the trigger is logged as `reason="geometry_failed"`. Both go
through `lifecycle_sweep._Sweep._unresolved` → the `[sweep] UNRESOLVED
(skipping) …` line the `/compare` skill greps. What is NOT graceful any more:
a parent cycle without a `CTS_ESTABLISHED`, a failed LOH map, a `BOS_CONFIRMED`
whose `confirmed_at` differs from its cycle's `CTS_ESTABLISHED.confirmed_at`
— those raise in `multitf/parent_tables.build_parent_tables` (see "Sub
Lifecycle-Start Clamp" below). Nor is the shared M15 FETCH (the input of every
sub, not one trigger): `multitf/data_bridge.fetch_lower_tf_data` retries a
failed OANDA chunk request twice, then raises, and an all-empty fetch
raises — never catch it into a skip (fixed 2026-09-27; in the 2026-09-22
audit a swallowed chunk error let a replay exit 0 on 2500 of 4228 M15
candles; `/compare` skill §2b).

---

## Event Sort Order Is a Dispatch Invariant

**Rule:** `sorted_events` in `_run_downstream_pipeline` is sorted by
`event_fields.processing_order_key` = `(ef.stamped_idx(e), e.type)` — the pre-E4
`(e.idx, e.type)`, pinned on the ANCHORS so the Plan E E4 flip (`ev.idx` :=
the moment on CTS_ESTABLISHED / BOS_CONFIRMED since E4a / E4b) reorders nothing (PLAN_E Q3; moment
order + an explicit type rank is post-Plan-E). The one WVMI tracker loop
(`_compute_wvmi_records` — the main and, since Plan G, every sub projection) walks this same
`sorted_events`. TIME walks sort on the MOMENT instead (`ef.event_moment`): zone_proximity's
threshold timeline (`(moment, type)`, Plan E E3g-2) and the POI sweep's CTS events
(stable, Plan E E3g-1); the wave-candle walk sorts on
`ef.cts_anchor_idx` (the same value on EST / UPDATED — a location walk). The alphabetical tie-break on
event type is **relied upon by handlers** — do not change the sort key without
auditing downstream dispatch logic. Pinned by `tests/test_event_order_pins.py`
(hazards H1–H3: BOS before EST of a cycle in every flip order; an EST before a
CTS_CONFIRMED on its moment; a CTS_THRESHOLD_UPDATED in [EST anchor, moment)
after the EST).

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
each snapshot's identity per `m15_df.attrs["sids"]`. **Plan C
(2026-09-20): that identity is the unique sub's `meta["sub_id"]`** (one
`SidRecord` per unique sub on the lens, `SidRecord.sub_id`; the pre-pool
tuple `(parent_sid, parent_cycle_id, sub_sid)` is gone — `parent_sid` /
`parent_cycle_id` in snapshot meta are informational, never identity, PART4
§17.9). The same applies to `export_chart_plotly`'s
M15-zone overlay — it reads zones from the M15.counter sub-entity in the
registry, never from `dfx.attrs["lower_tf_results"]`.

---

## Cross-entity sibling references require cadence-order interleaving (Session 3, 2026-05-31)

> **SUPERSEDED by Plan C (2026-09-20) — the sibling read is a POOL QUERY; the
> cadence chain is gone.** The rule below this banner is the current one; the
> Session-3 text after it is kept as history (it explains WHY the read must be
> causal, which has not changed).
>
> **Rule (PART4 §17.8).** The three sibling-referencing variations
> (`first_counter` / `subsequent_counter` read the confluence lens;
> `subsequent_confluence` reads the counter lens — `_SIBLING_LENS` in
> `multitf/entity_df_mutation.py`) resolve their reference zone with
> `_build_sibling_cts_ref_zone_from_pool(pool, other_lens, parent_sid,
> parent_cycle_id, probe_direction, idx_window=(lo, hi), m15_df)`:
>
> - candidates = `pool.records_for(other_lens, S, C)` filtered to
>   `r.direction == -probe_direction` (direction qualification, unchanged),
>   `not r.is_zero_length`, and LIVE somewhere in the window:
>   `r.start_idx <= hi and (r.trigger_end_idx is None or r.trigger_end_idx >= lo)`.
>   A record that exists but has not started (an FC record between its
>   `trigger_idx` and its `start_idx`) is excluded by `start_idx <= hi`.
> - each candidate's CTS events (`_CTS_EVENT_TYPES`) come from its sub's
>   geometry (`pool.get_by_id(r.sub_id).geometry` = `(bounded, slice_begin)`,
>   SLICE-LOCAL → shifted by `slice_begin`, deep-copied), **clipped to the
>   record's own live window ∩ `[lo, hi]`**: `r_lo = max(lo, r.start_idx)`,
>   `r_hi = min(hi, r.trigger_end_idx or hi)`. This clip is what reproduces
>   the pre-pool candidate set now that geometry runs to the DATA EDGE — a
>   REPLACED sub's later `CTS_UPDATED`s must not compete (sub `3304/−1` is
>   replaced at 3819 but its geometry continues; unclipped, its post-3819 CTS
>   events would move `subsequent_counter`(1,2)@4083's `starting_idx` 4027 —
>   a pool key both `4027/+1` records dedup on).
> - `hi` = the reading trigger's `trigger_idx` (`_sibling_cts_idx_window`;
>   `lo` = 0 for `first_counter`, the LOH of the prior sd-prox / CTS-prox
>   candle for `subsequent_*`). **The window is the ONLY thing keeping the
>   read causal** now that geometry is not bounded per trigger — never widen
>   it, never drop `hi`. The clip (per record) and the reference window both
>   key every CTS type (`structure/reference_zone._CTS_EVENT_TYPES`) on its
>   MOMENT (`ef.event_moment`, Plan E E3b, 2026-09-25): a CTS whose anchor is
>   `<= hi` but whose moment is after it is NOT a candidate (before E3b they
>   keyed on `ev.idx`, the anchor for `CTS_ESTABLISHED` / a pattern-path
>   `CTS_UPDATED`; 0 cells on the reference window). The recency pick keys on
>   the moment too. Any change moves `starting_idx` = pool keys → its own `/compare`.
> - the zone is built by `build_reference_zone_from_cts_event(events,
>   kl_zones=[], df=m15_df (the shared entity-absolute frame, not the winner's
>   slice-local `bounded.df`), sid=0, probe_direction, idx_window=(lo, hi))`.
>   `kl_zones=[]` is behaviour-preserving (GOTCHAS "`ref=cts_confirmed` in a
>   Replay Log Does NOT Mean…").
>
> **What replaces the cadence interleaving:** the sweep's phase-0 order key
> (`lifecycle_sweep._fire_order_key`): at one `trigger_idx`, `TRIGGER_FIRE`s
> run confluence before counter (`_LENS_RANK`), `first_*` before
> `subsequent_*` (`_TYPE_RANK`), then by `trigger_event_idx`; and
> `REVERSAL_SPAWN` sorts BEFORE `TRIGGER_FIRE` at the same idx (the
> reversal-born successor must exist before an H1 trigger at that candle
> runs its sibling read). Across different idxs the heap order is the cadence.
> A missing sibling still takes the resolver's own-frame ad-hoc BOS_0
> fallback (`_window_extreme_idx`, PART4 §4.3.4 step 4, unchanged; logged
> `[entity_compute] sibling-CTS unavailable …`), and a resolver that cannot
> build any zone returns a `ProbeFailure` → `UnresolvedTrigger(reason=
> "probe_failed")` — no longer a silently skipped sub. No `probe_failed` row
> appeared on the first Plan C replay (all four unresolved rows are
> `degenerate_parent_cycle`).
>
> **Expected behavioural delta (PART4 §17.8), stated as an expectation, not a
> measurement:** within a record's live window the sibling now sees the
> natural-end CTS stream (a `CTS_UPDATED` the old per-trigger bound had cut),
> so small `starting_idx` shifts are possible; each must be explained by such
> an event inside `[r_lo, r_hi]`. On the first Plan C replay every sub's
> `starting_idx` matched the predicted table (no such shift was observed).
>
> Retired with the chain: `build_two_entity_parent_cycle`, `_ChainCursor`,
> `build_parent_cycle_chain`, the two scratch M15 entity dfs and
> `_assert_m15_frames_aligned` (there is ONE shared M15 feature frame; the
> two lens dfs are copies of it, populated by the mirror — §17.9), and
> `_build_sibling_cts_ref_zone(sibling_entity_df, …)`.

**Symptom (if violated):** a sub probe that reads its sibling entity's CTS
resolves the reference zone to `None` (the sibling sid it needs isn't built
yet), silently falling back / skipping the sub. Subs vanish or shift; the
chain driver continues with no exception, so the regression manifests only
via `/compare` deltas.

**Root cause:** the reference-zone pivots made several probes read the SIBLING
entity (confluence ↔ counter): `first_counter` → sibling first_confluence's
CTS (Session 2); `subsequent_confluence` → sibling counter's CTS and
`subsequent_counter` → sibling confluence's CTS (Session 3). Each read must
land on a sibling sid that is already built. Because the two reads point in
OPPOSITE directions (confluence reads counter AND counter reads confluence),
no static "build entity A fully, then B" order can satisfy both — that is why
Session 2's confluence-first stopgap (`_run_first_confluence_multi_tf` before
`_run_multi_tf`) could only support `first_counter`.

**Rule:** the two M15 entities are built INTERLEAVED in trigger-cadence order
by `build_two_entity_parent_cycle` (driven by `_run_multi_tf_dual`), advancing
whichever `_ChainCursor` has the smaller next M15 trigger boundary. Cadence
guarantees every sibling CTS a probe reads is strictly EARLIER than the
reading sid's own trigger candle (`first_confluence`@BOS < `first_counter`@sd-prox
< `subsequent_confluence`@CTS-prox < `subsequent_counter`@next-sd-prox …), so
the needed sibling sid is always already built when the probe fires. Bootstrap
start resolution is therefore LAZY (deferred to the cursor's first `step`),
NOT eager at cursor construction.

**Frame-alignment invariant:** the sibling read is by entity-absolute M15 idx,
so the two M15 entity dfs MUST share one index frame (same candle ⇒ same idx).
Both are fetched over the same `(pair, M15, h1_start, h1_end)` range;
`_assert_m15_frames_aligned` (length + first/last `time`) fails loudly if a
future fetch-range drift breaks this.

**Why it's easy to reintroduce:** a refactor that reverts to building one
entity fully before the other, or that resolves a bootstrap probe eagerly at
cursor-construction time (before the sibling cadence has advanced), breaks the
cross-read silently. Guard: each sibling-reading resolver logs a WARNING when
the sibling CTS is missing.

**Direction qualification — the sibling can REVERSE before the trigger (2026-06-15).**
"Most recent sibling CTS" is NOT simply max-idx across all CTS: the sibling sub
may have reversed before this trigger fires, and a post-reversal CTS does not
represent a genuine confluence/counter relationship. `_build_sibling_cts_ref_zone`
filters candidates to `struct_direction == -probe_direction` (the sibling's own
bootstrap direction = `-lower_sd`) BEFORE the most-recent selection, and passes
that filtered list to `build_reference_zone_from_cts_event` (which re-selects by
idx — so filtering the local pre-selection alone is insufficient; the list handed
to the primitive must be filtered). *(Plan C: the same qualification is the
`r.direction == -probe_direction` filter on RECORDS in
`_build_sibling_cts_ref_zone_from_pool` — a record's `direction` is its sub's
absolute direction, and a reversal-born successor is a different sub with a
different record, so post-reversal CTS events never enter the candidate list.)* **Symptom if violated:** the counter (or
subsequent confluence/counter) sub starts far too late — anchored on the
sibling's still-sliding post-reversal `CTS_UPDATED` right before the trigger,
collapsing its probe window (`end_idx_reached`, iter=1) instead of rooting at the
sibling's last same-direction CTS. This also silently breaks the primitive's
`source_sd = -probe_direction` reconstruction (it assumes the picked CTS is from
a `-probe_direction` structure). Re-reversals back into the expected direction
re-qualify (most-recent qualifying wins). See PART4 §4.3 "Direction-qualified
sibling CTS".

---

## Subordinate `parent_extreme_dir` Must Use `-trigger.lower_sd`

**Rule:** Every parent→sub-TF probe-INPUT mapping MUST compute
`parent_extreme_dir = -trigger.lower_sd` (spec §4.3.1 unified rule).
Today one call site: `multitf/entity_df_mutation._resolve_first_confluence_via_unified_probe`
maps the parent BOS ANCHOR to the sub-TF `input_idx` for the unified probe (the
legacy `_resolve_via_legacy_probe` path is gone; `first_counter` / `subsequent_*`
take their input from the sibling lens, already on M15). It passes the value to
`map_candle_to_lower_tf(time, parent_extreme_dir, m15_df)` (signature
post-2026-05-29 rename from the older `mapping_sd` / `h1_sd` parameter — same
numeric semantics).

NOT input mappings, so NOT this rule (Plan E E5·3, 2026-09-25): the same
function's mapping of the parent CTS ANCHOR (`parent_cts_anchor_idx`, H1) into the M15 `probe_end_idx` uses `+lower_sd`
(the structure ceiling / floor), and the M15 chart's H1 zone-proximity markers
(display) pass the trigger wick's side. Do not "unify" either to `-lower_sd`.

**Why this is a landmine:** for `first_counter`, `lower_sd = -parent_sd`,
so `-lower_sd == parent_sd` — meaning earlier code (`mapping_sd =
trigger.parent_sd`) produced the correct result. The two expressions are
mathematically equivalent for counter and the test suite passes either
way.

But for `first_confluence` (3b+), `lower_sd = +parent_sd`, so the two
expressions diverge:
- `-lower_sd = -parent_sd` ✓ correct (BOS anchor: lowest low in bullish
  parent, highest high in bearish — the BOS candle's deepest touch of
  the broken zone, which is the OUTER of the confluence sub's reference)
- `parent_sd` ✗ wrong (would map to the parent BOS HIGH instead of LOW,
  shifting the confluence sub's input_idx off its proper anchor)

A future "simplification" back to `trigger.parent_sd` silently corrupts
every confluence (and any future variation where `lower_sd != -parent_sd`)
without breaking any existing test. Always keep the unified `-lower_sd`
form.

**Generalization to deeper nesting:** the same principle holds for
descendants beyond depth 1. Always express the mapping in terms of the
sub being built, never in terms of the parent.

---

## Sub WVMI Is Computed Inside the Projection, Ungated; `source_kinds` is a Return-Only Filter

Two related load-bearing invariants in `pipeline/orchestrator._run_downstream_pipeline`:
Rule 1 from Part 4 Step 3d.iii, Rule 2 from Plan G (2026-09-30, `plans/PLAN_G_wvmi_unique_sub.md`;
it INVERTS the 3d.iii rule "subs pass `skip_wvmi=True`, WVMI is parent-event-driven"). They look
like "cleanups that shouldn't change behavior" — they aren't.

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
"CTS")` returns None and every sub projection's WVMI produces 0 records. This was a
latent bug from Week 8 Part 3 (sub WVMI silently empty) — fixed in 3d.iii.

**Rule 2 — A sub's WVMI is computed INSIDE its projection, ungated (`wvmi="none"`).**

```python
# multitf/pooled_structure_build.project_to_window — ONE call per unique sub
downstream = _run_downstream_pipeline(
    bounded.df, clipped_events, direction,
    source_kinds=["BOS"], fib_mode="cross_cycle",
    structure_path_id=first_lens_path,
    wvmi="none",          # a sub: no gate, every CTS_CONFIRMED is offered to the tracker
    lifecycle_floor=floor, lifecycle_cap=cap, cap_reason=reason,
)
```

`wvmi` is ONE parameter, validated (`WVMI_MODES`; an unknown value raises — a typo must not
silently gate or drop sub WVMI): `"first_sd_prox"` (the default: the main — `check_zone_proximity`
runs and gates, see "WVMI Constraints" #4), `"none"` (a sub), `"off"` (tests of other stages). Every
mode that computes WVMI goes through ONE helper, `_compute_wvmi_records`: the tracker's frame is
`df.iloc[:cap + 1]` under a cap (the natural-end frame would let FP / LB / a lock LP read past the
sub's end), the temp LP stops at the cycle's `end − 1` and `cycle_collapsed` = `start >= end`, both
from the `compute_cycle_lifecycle` table KL / POI read. The mirror persists one deep copy per lens df
the sub is on, each with THAT lens's path in the FIELD and the meta; the orchestrator's post-pass
(`_stamp_sub_wvmi_trigger_meta`) then writes each lens copy's trigger metadata FIRST in its meta:

| Lens | `triggered_by_event_type` stream (§8.5's WVMI class) | `triggered_by_event_idx` |
|---|---|---|
| `H1.main >> M15.confluence` | sd-prox class: the main's first sd-prox per cycle (`ZONE_PROXIMITY_TRIGGER`) + each var 4 (`SUBSEQUENT_COUNTER_TRIGGER`) | the stream's FIRST entry (LOH-mapped) inside the sub's `[start_idx, m15_end_idx]`, in PARENT coords; None if none |
| `H1.main >> M15.counter` | CTS-prox class: each var 3 (`SUBSEQUENT_CONFLUENCE_TRIGGER`) | same rule on this lens's stream |

`parent_path_id` is always `"H1.main"` (the parent entity). On a sub row `triggered_by_*` is
**attribution, not a gate** (a declared rule-3 meaning change, "Event Contract Rules"): the record
exists whatever the stream says, and the trigger is often knowable only after the record (sub 3's
counter c0: created at M15 2752, its counter trigger H1 871 = M15 3487).

**Trap:** "the sub has no parent trigger in its window — skip its WVMI" or "gate the sub on the
first trigger again". Both restore the Plan C behaviour the user reversed (2026-09-28: "unique subs
are where trading decisions will be made and governed by life cycle, it would make sense WVMI lives
here as well"): a rendered sub gets WVMI like it gets zones. Equally, don't run
`check_zone_proximity` for a sub or unify the two paths' gates: the main keeps its first-sd gate,
and zone proximity also feeds the var3 / var4 detectors and the §8.5 streams (main only).

**No trigger in the window is a VALUE, not a skip.** A lens with no WVMI-class trigger inside the
sub's window writes `triggered_by_event_idx` / `_type` None (both keys present) — never "the
cycle's first var X" or another sub's trigger. The exporter writes the idx as nullable `Int64`
(an empty cell), so the other rows stay ints, and reads both keys STRICTLY (`meta[...]`, rule 3): a
record that reaches a CSV unstamped raises instead of passing for "no trigger".

**History (dated, Plan C → Plan G):** 3d.iii made sub WVMI parent-event-driven (`skip_wvmi=True`,
`multitf/sub_wvmi.compute_parent_driven_sub_wvmi` after the sub was built, a per-use-case gate
table); Plan C (2026-09-20) made it one trigger-gated sweep per unique sub, persisted into every
lens with the SWEEPING lens's path on the field; Plan G (2026-09-30) deleted `sub_wvmi.py` and
`persist_facade_wvmi_to_entity_df` and moved the computation into the projection.

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
| `var3_last_per_cycle` | `_run_multi_tf_dual` (orchestrator; was `_run_first_confluence_multi_tf`) | subsequent_confluence triggers |
| `var4_last_per_cycle` | `_run_multi_tf_dual` (orchestrator; was `_run_multi_tf`) | subsequent_counter triggers |

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

*(Plan C, 2026-09-20: this is now the state of the code, in pool terms —
`_run_multi_tf_dual` feeds EVERY var 3 / var 4 trigger to the sweep
("triggers are independent", PART4 §17.4); the "overwrite" is the
`same_dir_replacement` end on the incumbent `TriggerRecord` at the new
record's `start_idx` (§17.4 end condition 2), and `deactivated_by` does
not exist on any artifact.)*

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
| `meta["triggered_by_event_idx"]` | **Parent-df coords** (e.g. H1 for an M15 sub); None on a sub row whose lens has no trigger in the sub's window |
| `bos_structure_id`, `bos_cycle_id`, `locked_by_cycle_id` | Identity ints; coordinate-free |
| `cycle_collapsed` (field) | A flag, not an index (Plan G G2) |

**Why:** a sub record's trigger metadata is ATTRIBUTION to a parent event (Plan G G4 — see the
"Sub WVMI Is Computed Inside the Projection" landmine above): the lens's first WVMI-class trigger
(main first sd-prox, var 3, var 4) lives on the parent entity; the record's wave-candle indices
live on the sub entity. Both fields are ints called "_idx", but they index different dataframes.
(A `meta["lp_locked_at"]` row stood here until 2026-09-30 — no code has ever written that key.)

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

## Sub Slices Must Re-Derive `is_range_*` Labels After `reset_index`

**Rule:** In `multitf/entity_df_mutation._build_geometry` (the pool-free core
of `build_or_get_geometry`; was `build_one_sid` before Plan C), after the slice +
`reset_index(drop=True)`, the `is_range` / `is_range_confirm_idx` /
`is_range_lag` columns MUST be dropped and recomputed via
`apply_is_range_labels(trigger_df)` on the slice. The recompute is
REQUIRED, not optional.

**Why this is a landmine:**
`apply_is_range_labels` (in `patterns/range_label.py`) is called once in
`prepare_lower_tf_data` on the **entity-wide** M15 dataframe. It writes
`is_range_confirm_idx = t` where `t` is the **entity-absolute** positional
index of the confirm candle. Example: a candle whose confirm landed at
entity row 4049 gets `is_range_confirm_idx=4049`.

When `_build_geometry` builds a sub's natural-end structure, it slices that
entity_df and calls `reset_index(drop=True)`. **Row indices reset to
0..N, but column VALUES are unchanged** — so `is_range_confirm_idx` still
holds 4049 even though the corresponding slice-local position is, say, 72.

`MarketStructure.run()` then reads
`self.df.iloc[i].get("is_range_confirm_idx")` (in
`_is_range_candle_given_confirm` / `_finalize_range_candidate_offline`)
and **treats the value as slice-local**. Three cascading corruptions
follow:

1. **`RANGE_STARTED` / `STATE_CHANGED → RANGE` events emit at impossible
   idx.** `_finalize_range_candidate_offline` uses `confirm_idx` directly
   as the event idx. Then `mirror_lower_tf_result_to_entity_df` shifts
   every event by `slice_begin`, so e.g. confirm_idx=4049 →
   post-mirror idx=4049+3977=**8026** (far outside the M15 data range,
   ~4228 candles).

2. **Back-fill cap silently goes unbounded.** `_step_anchor` (line ~844):
   ```python
   min_d = D if confirm_idx is None else min(confirm_idx, D)
   for k in range(i, min_d):
       self._replay_step_no_patterns(k, freeze_range=True)
   ```
   With `D = i + range_max_k = i + 5` and polluted `confirm_idx` >> D, the
   `min` always picks D. Back-fill runs the full 5-candle window instead
   of stopping at the real confirm idx.

3. **CTS extreme leaks forward past the eventual pattern apply_idx.** The
   over-run back-fill keeps calling `_maybe_update_cts_pre_confirm(k)` on
   candles k > the eventual pullback apply_idx (cts_phase is still
   EST_OR_UPD during back-fill). `st.cts.idx` records a future extreme.
   When the pullback pattern then applies at apply_idx, `_emit_cts_confirmed_once`
   snapshots `cts_anchor_idx = st.cts.idx` — the **future-leaked** extreme.

**Visible symptom:** `BOS_{n+1}.idx` can land BEFORE `CTS_n.cts_anchor_idx`,
violating the cycle-progression invariant (each cycle should progress
forward in idx). On the chart this manifests as `BOS BOS CTS CTS` ordering
where you expect `BOS CTS BOS CTS`. Concrete reproduction case (counter
sub_sid=1 of parent_cycle=2 in the NZD_USD 2025-11→2026-01 window):
CTS_0 anchor leaked to idx 4050, BOS_1 picked idx 4048, visual
`BOS(4027)→BOS(4048)→CTS(4050)→CTS(4086)` — cycle_id sequence
`0→1→0→1` by chart-marker idx.

**Fix (codified in `_build_geometry`; was `build_one_sid`):**

```python
from engine_v2.patterns.range_label import apply_is_range_labels, RangeLabelConfig
trigger_df = trigger_df.drop(
    columns=["is_range", "is_range_confirm_idx", "is_range_lag"],
    errors="ignore",
)
trigger_df = apply_is_range_labels(trigger_df, RangeLabelConfig())
```

After the fix, `confirm_idx` is slice-local (small ints), all three
corruptions disappear: `RANGE_STARTED` emits at sensible idx, back-fill
cap works as intended, and CTS extreme stays bounded by the pattern apply
candle.

**Why H1 main is unaffected:** H1 main runs `apply_is_range_labels` and
`MarketStructure.run()` on the **same** dataframe — no slice, no
`reset_index`. Only subordinate subs trip this. The bug only became
discoverable once we added per-sub `structure_events.csv` debug exports
and inspected the events list for anomalies.

**Why this rarely surfaces visibly:** the visible cycle-progression-invariant
violation needs (a) a range candidate detected at an anchor where the
over-run back-fill reaches a NEW extreme, AND (b) a pullback pattern
detected on the NEXT anchor with apply_idx BEFORE that new extreme. Most
pullbacks fire after the price has clearly peaked, so the leak lands at
an idx earlier than the apply candle and is harmless. The bug lurks
silently for most data shapes.

**Related:** sibling of "Slice Copies Inherit Mirrored Structure Cols"
above — both are coord-system bugs introduced by the slice + reset_index
pattern. Any OTHER positional-index column added in future pre-processing
on the entity-wide M15 df must be similarly re-derived on the slice.

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

1. (History — REMOVED 2026-09-29.) `_rewind_to(jump_to)` replayed `from i = 0`, not from `start_idx`
   (hundreds-to-thousands of unrelated candles on an entity df), and carried four more hazards found by the
   Plan A/B audits: (a) the rebuild ignored nested jump requests, so a rebuilt prefix could differ from the first
   pass (the Plan B §2 "rebuilt-prefix" exception); (b) the MAIN H1 path would have replayed a sid >= 1 from candle 0;
   (c) the rebuild reset `structure_id` to 0; (d) the expiry's own `BOS_THRESHOLD_UPDATED(probe_no_break)` was wiped
   by `self.events = []` (the threshold jumped with no event). Its only caller was the reversal-watch expiry, which
   F3b made unreachable (MARKET_STRUCTURE_SPEC "A reversal confirming on E applies"); both were removed —
   MS no longer rewinds. The record: git `c36777c` and earlier.

2. `BreakoutPatterns(self.df, end_idx=effective_end)` precomputes / scans
   the full df up to the run's data edge (the upper bound landed with Plan A,
   2026-09-19 — item (b) below; the LOWER side is still unbounded). On a
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

> **Update (Path 2b, 2026-06-20):** point 5's `_write_df_row` `range_hi`
> `df.at` read is now an array read (`self._out["range_hi"][prev_i]`); see
> the batched-write landmine directly below.

---

## MS Batched Output Writes Must Seed From the df (Multi-Structure Chain)

**Rule:** Any change that batches `MarketStructure`'s per-candle output
writes into whole-column assignments (Path 2b) MUST initialize the output
arrays FROM the existing df column values, NOT from fresh defaults.

**Why:** `compute_structure` (H1 main) runs MULTIPLE structures in a loop,
**chaining them through the same df** — `df2` is reassigned to each
`ms.run()` output and fed as the input to the next structure. The original
per-row `self.df.at[row, col] = v` writes only ever touched THIS structure's
processed rows `[start_idx, end]`, leaving prior structures' rows intact in
the shared df. A naive whole-column flush (`self.df[col] = default_filled_array`)
writes DEFAULTS to every row outside this structure's range, **clobbering the
prior structure's rows back to nan/-1/""**. Symptom: H1 `final.csv` reverts
to defaults from `start_idx` onward, and H1 `kl_zones`/`wvmi` + the M15
entities cascade off the corrupted df — while H1 `structure_events.csv` stays
identical (events are unchanged; it's pure df-column corruption). Bounded
single-structure sub runs (M15) don't hit this directly (one structure per
df), but they cascade off the corrupted H1 df.

**Fix:** `_init_output_arrays` seeds each array via
`self.df[c].to_numpy(dtype).copy()`, so rows this structure never processes
flush back unchanged. Equivalent to schema defaults for the first/only
structure (the df then holds `_ensure_output_cols` defaults).

**Detection gap:** the small-df unit tests stayed green through this bug —
they don't exercise the multi-structure stitch. Only the full `/compare`
(21/21 byte-identical) caught it. Validate batched-write changes with a full
replay `/compare`, never tests alone. See memory
`project_ms_optimization_opportunity.md` UPDATE 2026-06-20c.

---

## A Back-fill Must Stop at a Reversal (FIXED 2026-09-28)

**Rule:** any loop that runs `_replay_step_no_patterns` offline over candles other than the one the step acts on
(a frozen back-fill — incl. `_post_apply_range_check`'s, which starts AT the apply candle) must end the step as soon
as the state is REVERSAL — the pending reversal is applied inside that call — and must not run a later apply /
finalize / re-step; and a per-candle emitter that can change the state or the CTS must skip in REVERSAL (the BOS
barrier, the raw CTS update and the proximity confirmation do — a reversal WINNER's apply candle is still stepped;
every reversal apply clears the pending too, so that step re-applies nothing).
Canonical: MARKET_STRUCTURE_SPEC "Reversal inside a back-fill".
**Why:** the run loop only checks REVERSAL between steps; inside one step the dead structure kept stepping, left
REVERSAL through an unguarded state setter and could reverse again (two `STATE_CHANGED(to=reversal)` per sid; the
H1 hand-off takes the first, `compute_reversal_idx_by_sid` the last). **Guard:** `_set_state` asserts nothing leaves
REVERSAL — a new setter / back-fill that forgets the stop crashes instead of corrupting (not caught by the sub
build's `except (ValueError, IndexError)`, so a hit ends the replay). (A second assert — the `_rewind_to` rebuild
never reaches a reversal — went with the rewind, 2026-09-29.) Measure a change here with `review_scripts/reversal_shadow.py` (every MS run: back-fill
applies, leaves, events after the terminal). Pins `tests/test_ms_reversal_terminal.py`.

---

## A Step Must Stop at Its Own Watch Expiry (FIXED 2026-09-29 — then REMOVED with the expiry, F3b)

**Status (2026-09-29):** F3b made the expiry unreachable (a pending confirming ON its watch's expiry candle applies),
so the stops below, the rewind and the run-loop guard were removed; the later-anchor CAP stays as a live rule (it still
shapes the step layout — MARKET_STRUCTURE_SPEC "Expiry inside a step"). What replaces the rule: a watch is never open
at its expiry candle (asserted in `_replay_step_no_patterns`). The entry is kept as the record of why they existed.

**Rule:** once `_maybe_expire_reversal_watch` has requested a rewind (`jump_to_idx` set) inside a step, the step
ends at that candle and returns the jump target — no winner apply, no range finalize, no re-step (the winner
back-fill, after the apply, the range back-fill, `_post_apply_range_check`) — and a reversal candidate against an
open watch's frozen barrier must apply by the watch's `expires_idx` (`_best_bopb_pattern_at_anchor` caps it like
the scheduler; apply == `expires_idx` stays a winner). Canonical: MARKET_STRUCTURE_SPEC "Expiry inside a step".
**Why:** the run loop honours the rewind only between steps, and the rewind + seed restore (a direct
`state.state =`) throws away whatever the step did after the expiry — silently, a REVERSAL included (zones-audit F3;
200 rewinds entered in REVERSAL on 12k random tails, 0 on the window and the suite). And not everything was thrown
away: a second expiry in the continuation OVERWROTE `jump_to_idx` + the seed (wrong rewind target → df invariant
"bos_threshold changed during reversal watch"), and a `_rewind_to` rebuild replayed the continuation (a re-applied
reversal crashed its assert). **Return the jump target, not `k + 1`:** a rebuild ignores the request but resumes at
the returned index ("Deep-Couples…" 1(a) below); `k + 1` changed the rebuilt prefix and
`test_ms_stop_after_cts::test_mechanism` then raised `[INV] bos_threshold changed during reversal watch`. **Guard:** `run()` asserts a rewind is never
requested in REVERSAL (an AssertionError, not caught by the sub build's `except (ValueError, IndexError)`). Measure
with `review_scripts/reversal_shadow.py` (`expiry` / `win_past_exp` / `post_expiry` / `rewind_in_rev`). Pins
`tests/test_ms_expiry_stop.py`. Back-to-back watches (an expiry's rewind to anchor + 1 opening a new watch there) are two watches: df invariant 4
compares one watch's rows only (same frozen barrier, 2026-09-29; before, they tripped it and ended the run).
**Since F3b (2026-09-29) no expiry fires:** a pending reversal confirming ON the expiry candle — the only way an
expiry was reached — now applies (MARKET_STRUCTURE_SPEC "A reversal confirming on E applies"); the stops, the rewind
and this entry's guard were removed (the status above). `tests/test_ms_expiry_stop.py` is gone: its boundary pin and
the cap stub live in `tests/test_ms_reversal_on_expiry.py`.

---

## A Watch Guards ONE BOS — a New Cycle Ends It (2026-09-29)

**Rule:** a reversal watch freezes the current cycle's BOS; anything that replaces that BOS while the watch is open
must END the watch (drop its pending reversal) — today the only such write is a new cycle's `BOS_CONFIRMED`
(`_end_watch_superseded_by_new_cycle`, recorded as `meta["ended_watch_pattern_anchor_idx"]`). Every other
`bos_threshold` write either skips during a watch (`_bos_barrier_step`) or clears the watch in the same call
(`rv_anchor_failed`; until 2026-09-29 also the expiry and the rewind's seed restore — removed, F3b). A new write site
that moves the BOS under an open watch brings back what this closed.
**Why:** a watch left open across a new cycle applied its pending reversal on the superseded barrier — sometimes on a
close that never broke the NEW BOS — and df invariant 4 raised, ending the replay (AssertionError, not caught by the
sub build's `except (ValueError, IndexError)`); or the old watch's expiry rewound the new cycle away (no expiry
exists since F3b). **No exception:**
a pending confirming ON the establishing candle is dropped too — kept, it reversed the new cycle on the superseded
barrier on the same candle, with a close that cannot be beyond the new BOS (decided after the landing review of
`2285232`). The cycle-0 `BOS_CONFIRMED` needs no call: a watch needs a BOS, and there is none before cycle 0. **Guard:** df
invariant 4 (now a pure tripwire); pins `tests/test_ms_new_cycle_ends_watch.py`; measure with
`review_scripts/reversal_shadow.py` (`in_watch_est`) and `random_tail_search.py` (`iwe`). MARKET_STRUCTURE_SPEC
"A new cycle ends an open watch".

---

## `MarketStructure.run()` Crashed When `start_idx >= n` — RESOLVED (Plan E E1b, 2026-09-24)

**Status:** FIXED. The early return now builds the levels like the normal path
(`self._events_to_structure_levels()`, `[]` there), so `run()` with a `start_idx`
at or past the end of its working df returns `(df, [], [])`. Pin:
`tests/test_ms_stop_after_cts.py::test_run_with_start_past_the_frame_returns_empty_levels`.
What follows is the record of the bug; the remaining caution is semantic — an
empty result from a past-the-frame start is not a structure, so callers that
stitch successors (reversal handoffs) should still treat it as "no successor".

**The bug (before E1b):**

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

> **Plan C (2026-09-20) — ownership is the lifecycle window, per direction
> (PART4 §17.9 / §16.5 rev 2).** The map is now
> `owner_by_idx_dir[(candle, direction)]`, built by
> `export_m15_chart._compute_owner_by_idx_dir(sid_records, edge_idx)`: each
> sub `SidRecord` (identity = `sub_id`, via `_sid_record_identity`) claims
> `[start_idx, end_event_idx or edge_idx]` — its REAL-TIME lifecycle window
> (`SidRecord.start_idx` / `end_event_idx = sub.end_idx`), NOT
> `creation_event_idx` (= `starting_idx`, the structural anchor) — under the
> key `(candle, starting_sd)`. Rows are walked in `(start_idx, sub_id)` order
> so a later start wins a same-direction overlap; opposite-direction subs
> never collide (a `+1` and a `−1` sub may both be live on one chart, e.g.
> `4027/+1` on confluence from 4083 while `3760/−1` runs to 4200). The
> per-element check is `_owned_here(idx)`: **the default-keep heuristic below
> is RETIRED** — a candle nobody owns draws nothing. **Chart review
> 2026-09-20 refinement (option 2): two layers.** `_is_live(idx) = idx >=
> sub.start_idx` picks the layer: a live candle is drawn iff the LIVE map
> (`_compute_owner_by_idx_dir`, later start wins) names this sub; a candle in
> the sub's FORMING span `[starting_idx, start_idx)` is drawn iff the
> FORMING map (`_compute_forming_by_idx_dir`, later anchor wins) names it —
> independently of any live sub of the same direction — in the dotted/dimmed
> forming style. Do NOT "simplify" this back to one map: hiding the forming
> span broke every BOS→CTS line mid-structure (the first Plan C chart), and
> letting the live map veto forming elements left `3304/−1` / `3760/−1` as
> single points. The rendered-vs-emission idx rule in the table below is
> unchanged. Persisting elements (KL / POI rectangles, fibs) are NOT filtered
> by these maps: drawn from their anchor, active from `start_idx` (the KL/POI
> clamp) — except collapsed-cycle zones, which both charts skip
> (`_zone_render.is_collapsed_cycle_zone`; CHARTING_SPEC "Collapsed-cycle").
>
> **Chart review 2026-09-22 — ownership decides EXISTENCE, not STYLE.** Both
> maps still decide which points a sub draws at all; solid-vs-dotted is now a
> separate, purely visual rule (`_prior_line_segments`, CHARTING_SPEC "Recent vs
> prior"): a line SEGMENT is dotted iff a structure with a higher
> `_recency_key` `(parent_sid, parent_cycle_id, sub_id)` draws a segment over
> the same candles (>1 shared candle, direction-agnostic, per lens, whole
> segments), and is solid otherwise — **including structures that were never
> live in real time**. So `phase` (real-time) and `layer` (recency) are
> independent and both ride in the hover; do NOT re-derive one from the other,
> and do NOT expect a dotted line to mean "not tradeable" — an ACTIVE zone can
> hang on a dotted segment once a later structure supersedes it (accepted at
> review; the zones carry the real-time lifecycle). Consequences: the styles are
> `structure.m15.*_prior` (renamed from `*_forming`); the polyline is drawn as
> one trace per run of same-styled segments (`_group_flag_runs`), so a
> style-only change shows up in a chart census as renamed traces; every sub's
> polyline must be built BEFORE any is drawn (the pre-pass
> `_build_sub_polylines` + `sub_ctx`), because the rule compares across subs.
> Two adjuncts: the H1 OVERLAY on the sub charts is lifecycle-FILTERED per wave
> (`_wave_touches_window` / `_split_polyline_by_wave` — a wave never inside its
> H1 sid's `[first anchor or reversal handoff, reversal]` window — a LOCATION start,
> `_h1_overlay_window_start_by_sid`, not the moment-based struct_start (Plan E E3f
> decision) — is not drawn; the H1 chart itself is
> untouched and still draws every sid), and a sub ended by
> `same_dir_replacement` breaks its final segment at
> `_replacement_break_point` (the counter-move extreme up to the REPLACING
> structure's anchor — the structural swing, not the literal extreme).

**Rule:** §13.5.c.iii implements spec §16.5's "most recent sid only per
candle" rule for sid-tied display elements (CTS dots, BOS markers, swing
lines, PB markers, prev_bos lines, wave-candle hover anchors) by checking
`owner_by_idx[rendered_candle_idx] == this_sid_identity` (the identity
tuple `(parent_sid, parent_cycle_id, sub_sid)` before Plan C; `sub_id` now —
see the banner). The check uses the
**rendered candle's idx**, NOT the event's emission idx, because for some
events these differ:

| Element | Rendered candle |
|---|---|
| CTS_CONFIRMED dot | `meta["cts_anchor_idx"]` (the anchor), not `ev.idx` (the confirmation candle) |
| BOS_CONFIRMED dot | `ef.bos_anchor_idx(ev)` (the anchor), not `ev.idx` (the moment since Plan E E4b) |
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

**`owner_by_idx` construction (pre-Plan C, history):** walk SidRecords sorted by identity tuple
`(parent_sid, parent_cycle_id, sub_sid)` asc; for each, claim
`[creation_event_idx, end_event_idx]`. Later sids overwrite earlier in the
dict — but since merge-and-bound sids are sequential & non-overlapping,
ranges don't actually overlap, so `owner_by_idx[i]` is just the single
owning sid. Built once per chart export by `_compute_owner_by_idx`.
*(Plan C: `_compute_owner_by_idx_dir`, `[start_idx, end_event_idx or edge]`,
keyed per direction, `(start_idx, sub_id)` walk order — see the banner.)*

**The default-keep heuristic (pre-Plan C, RETIRED):** `owner_by_idx.get(idx, eid) == eid` —
when a candle isn't covered by any SidRecord (e.g., outside every sub's
lifecycle), default to "owned by this sid" so the rendering doesn't
disappear. Practically rare today (most rendered idx fall inside some
sid's range), but the default is safer than skipping. *(Plan C: unowned
candles draw nothing — hiding pre-`start_idx` dots is the point.)*

**Don't fall back to `ev.idx` filtering uniformly** — it loses the
cts_anchor_idx case. If you find yourself adding a new event-sourced
rendering, decide which idx to filter on: the emission idx or the
rendered idx.

---

## Mirror Translation of Nested-Dict Idx Fields Hardcodes Key Names

> **STILL LIVE (correction, 2026-05-27).** An earlier note here predicted the
> mirror would be removed by the merge-and-bound redesign "which builds directly
> in entity coords." That did NOT happen: `build_one_sid` still runs
> `compute_bounded_structure` on a sliced+reset_index df and calls
> `mirror_lower_tf_result_to_entity_df` (entity_df_mutation.py ~712) to shift
> slice-local → entity-absolute. So this translation is load-bearing and growing
> (FibState lifecycle fields were just added to it). Treat the entry as current.
>
> **Still live under Plan C (2026-09-20).** `build_one_sid` is gone but the
> shape is the same: `_build_geometry` runs `compute_bounded_structure` on the
> sliced + `reset_index` df (geometry is SLICE-LOCAL + `slice_begin`, stored on
> `PooledStructure.geometry`), `render_sub_projection` derives the sub's
> elements on that slice, and `mirror_lower_tf_result_to_entity_df` translates
> once per lens df the sub is on (`sub.lenses()`). Two more places now shift
> slice-local → entity-absolute and must follow the same key lists:
> `_build_sibling_cts_ref_zone_from_pool` (shifts the sibling's CTS events by
> `slice_begin` with `_EVENT_META_IDX_KEYS`) and `_resolve_reversal_start`
> (shifts the reversal probe's `starting_idx` / `finalize_idx` / cache key).

**Rule:** `mirror_lower_tf_result_to_entity_df` in
`multitf/entity_df_mutation.py` shifts slice-local idx → entity-absolute
by adding `slice_begin`. Top-level meta keys are driven by the tuple
constants `_EVENT_META_IDX_KEYS` / `_ZONE_META_IDX_KEYS` /
`_FIB_META_IDX_KEYS` / `_WAVE_CANDLE_META_IDX_KEYS` (single source of
truth; since Post-E·2, 2026-09-26, EVERY index-valued top-level meta key
is listed — until then 15 keys were exported slice-local, PLAN_E §9.3).
**Nested-dict idx fields, however, are translated by
per-record special-case loops that hardcode the key name string** — and
each loop must match exactly one producer-side key.

Current nested-dict idx fields and their hardcoded loop keys:

| Producer site | Nested dict | Key in producer | Mirror loop key |
|---|---|---|---|
| `zones/kl_zones_v1.py` (INIT + expansion) | `meta["bounds_steps"][k]` | `"start_idx"` | `"start_idx"` |
| `zones/poi_zones.py` `_compute_poi_activation_history` | `meta["activation_history"][k]` | `"idx"` | `"idx"` |
| `zones/kl_zones_v1.py` `derive_kl_zones_v1` (lifecycle finalize) | `meta["activation_history"][k]` | `"idx"` | `"idx"` |

(`FibState.activation_history` is gone — FIB_LIFECYCLE_SPEC §15.2; the row that
listed it was removed in the Post-E·2 landing review, 2026-09-26.)

**Also note (FibState):** `bos_idx` / `cts_idx` / `start_idx` / `end_idx` and each
`cts_history` `(idx, price)` entry are top-level dataclass fields (not meta), so
the mirror shifts them directly in the fib `replace(...)` call — NOT via the
meta tuple constants (`cts_history` since the Post-E·2 landing review; attrs
only). `end_reason`/`status` are not indices and pass through unshifted. The
lifecycle cap reaches FibTracker slice-local (`lifecycle_cap`, via
`project_to_window` → `_finalize_lifecycle_fields`), exactly like the KL / POI
caps, and this shift makes `end_idx` entity-absolute. Every shift site is
pinned on a synthetic result (`tests/test_event_meta_idx_keys.py::
test_mirror_shifts_every_site_exactly_once`) — before that the fib fields,
the wave-candle fields and `prev_bos_lines` had no test at all.

**Hazard:** if the producer renames its key (or adds a new idx-bearing
key in a nested dict), and the mirror loop isn't updated, the special
case silently no-ops. Slice-local values get persisted as if they were
entity-absolute. The chart consumer reads them via `_lt_time(idx)` which
expects entity-absolute coords (`charting/export_m15_chart.py` `_lt_time`),
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
MarketStructure's cycle-0 snapshot passes `direction=sd` (POI IC
validation in `find_ic_candidates` is strict by design too). No production
call is permissive (the unfiltered `get_unfilled_imbalances` /
`has_imbalance_in_range` were deleted 2026-09-30 — they had no production
caller).

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

**Sites involved** (by function — line numbers drift; per-site table with
window / `check_to_idx` / `evaluated_at`: IMBALANCE_FILL_SEMANTICS.md
"Consumer call-site matrix"):
- `zones/fib_tracker.py` — the `FibTracker._has_unfilled` helper (every
  decision read: `_on_cts_established`, the cross_cycle cycle-0 first
  activation on update, `_c0_has_unfilled_now`, `_update_fib_cts`,
  `_update_cycle1_main`) + the two uncut cycle-0 cache writes
  (`_handle_sid1plus_cts_established`, `_handle_cycle0_cts_updated`)
- `zones/cross_cycle_fib.py::resolve_cross_cycle_eligibility` — the own test
  + both dead-cycle walks (`direction = sd` when `sd ∈ {+1, -1}`); reached via
  `select_fib_anchor_for_cycle`, `_maybe_activate_main_cross`, `_m15_cross_check`
- `structure/market_structure.py::_update_cycle0_data`

Plus `select_fib_anchor_for_cycle` takes `struct_direction` as a
parameter; the two callers (`compute_poi_inners_for_cycle` and
FibTracker's internal use) must pass it. Default value of 0 falls
back to permissive — kept for backward compat but no production caller
should hit it.

**The second knob — `evaluated_at` (Plan F, 2026-09-24):** the same sites
also pass the MOMENT of the question. `has_unfilled_imbalance`,
`resolve_cross_cycle_eligibility` and `select_fib_anchor_for_cycle` take it
keyword-only and REQUIRED, so a forgotten moment is a `TypeError`, never a
silent default; `evaluated_at=None` is an explicit, justified no-cut
(FibTracker's decisions pass the handled event's moment, `event_moment(ev)` —
the apply candle `confirmed_at` on a pattern-path CTS_UPDATED since Plan E E3·0; its two cycle-0 cache
writes, `_update_cycle0_data` and `compute_poi_inners_for_cycle` pass `None`).
Rule: IMBALANCE_FILL_SEMANTICS.md "Knowability — the c3 rule".

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
(As of 2026-05-26. Since Plan F, 2026-09-24, the re-activations wait for the
gap's first c3 — 953→954, 997→998, so `confirmed_idx` is 998; `[905,951]` is
unchanged.) Full causal chain in GOTCHAS "POI `confirmed_idx` Is a Lossy Scalar".

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

---

## FibState Lifecycle Gate Is Per-Record, NOT Cycle-`status` (Session 2)

**Rule:** After the Session-2 lifecycle migration (`FIB_LIFECYCLE_SPEC.md`),
`FibState.active` is **condition-only** and `status` is a **cycle-level** label
shared by every version record of a cycle. The consumer gates therefore use
**raw per-record `active`/`locked` for version distinction** and `status` (or
`end_idx`) ONLY for the cycle-level ended/disappeared terminal:

- **POI gate** (`zones/poi_zones.py`): `(fib.active AND fib.end_idx is None) OR fib.locked`
- **Chart gate** (`charting/export_plotly.py`): `status != "disappeared" AND ( locked OR (active AND status not in {"ended","disappeared"}) )`

**Do NOT "simplify" the POI gate to `status == "active" OR locked`.** It looks
equivalent and the spec's first draft proposed it, but it is a **silent
byte-identical regression on subordinate structures**: a superseded (dead) cross
version of a *still-live* cycle carries per-record `active=False` yet inherits
the cycle-shared `status="active"`, so the cycle-status gate would wrongly
*process* it → extra/incorrect sub POIs. Fibs are in no CSV and M15 sub charts
have fib lines off, so this would NOT show in a `/compare` CSV/PNG diff — it
leaks only into rendered sub POI zones. Same reason the chart gate (§9.1) is
per-record.

**Do NOT make `_deactivate_cross` (`cross_failed`/`cross_shortened`) stop setting
`active=False`.** Unlike the cycle TERMINALS (new_cycle / scenario1_revert /
reversal / lifecycle_end — which now set `end_idx`/`end_reason` and leave
`active` alone; under Plan C the cap-produced terminal carries the sub's
`end_reason` — `parent_end` / `same_dir_replacement` / `reversal` — in place
of `lifecycle_end`, same path), these are **version-internal supersedes**: the cycle stays
alive via the next version / single fallback, but the dead version MUST read
`active=False` so the per-record chart gate drops it (the "dead-version trail"
vanish). They deliberately do NOT call `_set_terminal`.

**Why the §9.2 visible vanish is latent (don't mistake it for a no-op):** in the
default config the migration is byte-identical on CSVs + PNGs because (a)
H1-main fibs almost always lock (no unlocked-inactive/ended H1 fib to vanish),
(b) M15 sub charts render no fibs (`"fib": {"lines": False}`), (c) fibs are in
no CSV. To actually SEE dead-version / ended-unlocked fibs vanish, enable M15
fib lines. The migration still ran (H1 fib hover labels switched to the
`status`-based form, e.g. "ended (locked)").

---

## Sub Lifecycle-Start Clamp: Uniform, Parent-Floored; Start AND End Resolution Are Shared

Two interlocking facts about the cycle/structure lifecycle clamp (B1 start +
B2 Phase A end, 2026-05-27; **rewritten for Plan C, 2026-09-20** — the floor
now lives ON THE RECORD, PART4 §17.4). Canonical model + H1→M15 mapping live
in `PART4_REFACTOR_SPEC.md §5` / §17.4–§17.6; session writeup in
`memory/project_cycle_lifecycle_parent_cycle_floor.md`; the 3.2b failure that
forced the floor onto the record is GOTCHAS "A Lifecycle Floor That Lives in a
Build Function Is Lost by Any Path That Bypasses It".

**1. The clamp is UNIFORM — no per-cycle exception.** A zone's first-active is
`max(its own confirmed_idx, its structure's lifecycle-start)`, where the
structure's lifecycle-start embeds the parent floors for subs. It is applied at
the **structure level** (`struct_start_by_sid[sid]`, raised by `lifecycle_floor`)
so it lands on **every cycle of a structure including cycle 0**. There is NO
main-style "cycle 0 == struct start, skip the clamp" shortcut on subs — a sub
cycle 0 must still floor at the parent floor. Do not add a per-cycle carve-out.

- **The floor is a FIELD of the record, set once at creation (Plan C).**
  `lifecycle_sweep._Sweep._resolve_and_record` (step 5) creates every
  `TriggerRecord` with
  `start_idx = max(probe_finalize_idx, trigger_idx, parent_floor_idx)` where
  `parent_floor_idx = ParentTables.floor(S, C)` = `floor_m15[(S,C)]` =
  `LOH(max(struct_start[S], cts_moment[(S,C)]))` (`multitf/parent_tables.py`;
  `LOH` = `_map_parent_idx_to_m15_hour_end`, last-of-hour — the timing mapper).
  All three terms are load-bearing (§17.4): `probe_finalize_idx` when the probe
  finished after the trigger (FC(0,0): trigger 463 → start 1020),
  `trigger_idx` when the structure was already known before this trigger
  fired (a probe-cache hit inherits an earlier finalize — see "Probe Cache
  Keys Are Shared by Reversal Handoffs"), `parent_floor_idx` when the parent
  cycle was not alive yet. Historical fields (`starting_idx`, `trigger_idx`,
  `probe_finalize_idx`) are never adjusted — only `start_idx` is. The unique
  sub's `start_idx` = its FIRST non-zero-length record's `start_idx` (§17.5,
  sweep phase 2). `build_parent_cycle_chain`, `build_one_sid` and its local
  `_floor_abs` are gone; there is no per-consumer re-derivation of the floor.
- **What the zone layer receives.** `render_sub_projection` passes the unique
  sub's `start_idx - slice_begin` as `lifecycle_floor` (and `end_idx -
  slice_begin` as `lifecycle_cap`, `end_reason` as `cap_reason`) to
  `project_to_window` → `_run_downstream_pipeline` → `kl_zones_v1` /
  `poi_zones` / `fib_tracker` → `compute_struct_start_by_sid` /
  `compute_cycle_lifecycle`. The zone derivations stay parent-agnostic — they
  receive ONE `lifecycle_floor` int. Don't thread `parent_sid` /
  `parent_cycle_id` into `kl_zones_v1` / `poi_zones` (they're shared with the
  main entity, which has no parent). There is exactly one projection per
  unique sub, mirrored into every lens df it belongs to (§17.9) — so both
  charts see the same floored window.
- **Lifecycle-only.** The clamp moves ONLY `confirmed_idx` / `activation_history` /
  fill. It must NOT move `base_idx`, the rendered rectangle outline
  (`start_time = _time(base_idx)`), the MS run / BOS / CTS, `starting_idx` (the
  pool key; `SidRecord.creation_event_idx` for a sub), `trigger_idx` or
  `probe_finalize_idx`. A `/compare` showing any of those shifted is a red
  flag. (Pre-Plan C this list named `start_trigger_idx`, "the sub-WVMI gating
  window"; that field is split into `trigger_idx` / `probe_finalize_idx` /
  `start_idx`, and the sub-WVMI trigger-attribution window is now
  `[sub.start_idx, sub.end_idx or edge]` — the floored lifecycle itself; since
  Plan G the sub's WVMI itself is computed in its projection over that same
  floor / cap, ungated.)
- **Former graceful-degradation paths are now ASSERTS.** Before Plan C a sub
  whose `(parent_sid, parent_cycle_id)` was absent from the floor dict, or
  whose H1→M15 map returned `None`, silently lost the parent term. Now
  `build_parent_tables` raises when a `CTS_ESTABLISHED` lacks
  `meta["confirmed_at"]`, when a `BOS_CONFIRMED (S,C)` has no `CTS_ESTABLISHED`,
  when `BOS_CONFIRMED(S,C).meta["confirmed_at"] !=
  CTS_ESTABLISHED(S,C).meta["confirmed_at"]` (the definitional identity —
  NEVER assert it against the CTS anchor `meta["cts_anchor_idx"]`, a location;
  `CTS_ESTABLISHED.idx` is the moment since Plan E E4a), and when any LOH
  map returns `None`; the sweep asserts `tables.has_cycle(S, C)` for every
  trigger it resolves and `parent_sd[S]` for every record it creates. Nothing degrades to an
  unfloored sub. The one remaining zone-layer soft spot — a zone with no
  `structure_id` gets no clamp — cannot fire for subs (the internal
  `structure_id` is always 0 inside a sub run).

**2. Start AND end resolution are SHARED (B2 Phase A, 2026-05-27).** All three
helpers live in the pure-leaf `zones/structure_lifecycle.py`, called by
`kl_zones_v1`, `poi_zones`, `fib_tracker` AND (Plan C) `multitf/parent_tables.py`
(which uses `compute_reversal_idx_by_sid` + `compute_struct_start_by_sid` and
re-states the cycle-start/end rule of `compute_cycle_lifecycle` on H1 → M15):
- `compute_struct_start_by_sid` — per-`structure_id` lifecycle-start (the clamp): the
  structure's first `CTS_ESTABLISHED` MOMENT (Plan E E3f, 2026-09-25; before it the
  first anchor, BOS_0's — e.g. H1 sid 0 96 → 115), then the reversal handoff and floor.
- `compute_cycle_lifecycle(events, reversal_dict, floor, cap, cap_reason)` —
  per-`(sid, cycle)` `(start, end, end_reason)`. **Start = the CTS-established
  MOMENT** (`CTS_ESTABLISHED.meta["confirmed_at"]`, the apply candle — `== ev.idx`
  since Plan E E4a; Plan C 2026-09-20 — NOT the CTS anchor `meta["cts_anchor_idx"]`,
  the pattern's extreme candle, a historical location like the BOS anchor), clamped by
  `struct_start` and the floor. **End is a pass-through:**
  `end = min(next-cycle clamped start, reversal, lifecycle_cap)`, never computed
  per-zone. KL/POI/fib INHERIT `end_idx` / `end_reason` from this table.
- `compute_reversal_idx_by_sid` — the single reversal dict (the old duplicated
  `_get_reversal_confirmed_by_sid_from_events` / inline `reversal_idx_by_sid`
  are gone; since 2026-09-27 the chart's `export_plotly._get_reversal_confirmed_by_sid`
  delegates to it, and the orchestrator's `reversal_idx_by_new_sid` — FibTracker's
  Scenario-1 argument, its reversal terminals, the prev-BOS line END — is it shifted
  to the new sid; it used to be the last `REVERSAL_CANDIDATE.meta["apply_idx"]`, a
  SCHEDULED apply that may never realise (then, a watch expiry could discard it), which put a phantom 'reversal' on
  the fibs: `tests/test_orchestrator_reversal_source.py`). Since 2026-09-28 the
  zone-proximity scan cap (a DISCARDED candidate that was a sid's last skipped every
  later cycle's scan: no triggers, no WVMI gate record, no M15 subs; 0 on the
  reference window) and the H1-main `SidRecord.end_event_idx` read it too
  (`tests/test_zone_proximity_reversal_cap.py`) — no timing/lifecycle reader of the
  scheduled apply is left (the chart's candidate markers show the candidate's anchor /
  pattern).

**Any change to end resolution or the reversal-dict construction goes in the
helper, NOT per-zone** — the whole point of the pass-through is one source of
truth (the drift hazard from [[feedback-in-flight-vs-downstream-resolver]]).
Subs pass `lifecycle_cap` (the mirror of `lifecycle_floor` — `min` for end vs
`max` for start) into the derivation. **The cap is no longer the run bound**
(Plan C): the sub's geometry runs to the DATA EDGE (`build_or_get_geometry`,
`run_cap_abs = len(m15) - 1`) and the cap is purely the projection's
lifecycle input — see "Lower-TF Zones, POIs, and Fibs Must Be Capped at
Lifecycle End".

**The floor and the end-cap come from ONE table, on the MOMENT — so they
cannot diverge (Plan C, supersedes B2 Phase B).** `multitf/parent_tables.py`:
`cts_moment[(S,C)] = CTS_ESTABLISHED.meta["confirmed_at"]` (last-seen per
cycle; `== BOS_CONFIRMED.confirmed_at`, asserted), `floor_h1[(S,C)] =
max(struct_start[S], cts_moment[(S,C)])` (the cycle's CLAMPED
lifecycle-start), `end_h1[(S,C)] = floor_h1[(S,C+1)]` if that cycle exists,
else `rev_by_sid[S]` (`STATE_CHANGED→reversal`, never `REVERSAL_CANDIDATE`),
else `None`; both LOH-mapped to `floor_m15` / `end_m15`. The record's
`parent_floor_idx` is `floor_m15[(S,C)]` and its `parent_end` candidate is
`end_m15[(S,C)]` (sweep step 6) — by construction the end of `(S,C)` IS the
floor of `(S,C+1)`. The B2 Phase B rule that stood here ("next-cycle source
= `CTS_ESTABLISHED.ev.idx`, don't revert it to `confirmed_at`") had the right
aim (floor and cap from the same field) and the wrong field: the anchor is
historical, the moment is when the cycle became tradeable, and the two differ
whenever the CTS anchor precedes the apply candle (on the reference window 3
of the 34 `CTS_ESTABLISHED` CSV rows = 2 unique M15 sub cycles, sub 3 being
mirrored into both lenses, each with an empirical lag of 1 candle; anchor ==
apply candle is the COMMON case — 31/34, including all five H1 cycles — not
luck. The only bound is `pattern_anchor_idx <= cts_anchor_idx <= confirmed_at <= pattern_anchor_idx + 5`;
ARCHITECTURE "`ev.idx` convention"). Retired with
it: the 4 trigger detectors' `lifecycle_end_idx` (the unread field was deleted
in Plan E E1b), `_find_m15_lifecycle_end`, `parent_end_lookup`, `parent_struct_end_m15`
and `run_pipeline`'s `parent_cycle_floor_h1`. Same rule inside a sub:
`compute_cycle_lifecycle` floors each sub cycle on ITS `CTS_ESTABLISHED`
moment, so main, sub cycles and the parent table agree (expected from the plan:
three sub-cycle starts +1 — 1223→1224, 2828→2829 ×2, i.e. 3 lens rows = 2
unique cycles, the 2828→2829 one being sub `2639/−1` mirrored into both lenses;
measured on the first Plan C replay: one visible shift, sub `454/+1`'s cycle-1
end 1223→1224 — the 2828→2829 pair is masked by an equal floor, GOTCHAS "A
Predicted +1 Shift Can Be Masked by an Equal Floor"; H1 byte-identical).
Collapsed cycles (clamped `start >= end`) are
uniformly `status="inactive"` with empty `activation_history` (outline-only) —
this replaced the prior split
where the inner KL derivation said `"ended"` and the cap said `"inactive"`.
Under Plan C a DEGENERATE parent cycle (`floor_m15 >= end_m15`) never reaches
the zone layer at all — its triggers are unresolved (see "Degenerate Parent
Cycles" below).

---

## Sub-Structure Pool: Alignment Is Direction-Derived, NOT Sticky-Per-Entity (Phase 2, 2026-07-08)

> Design LANDMINES for the sub-structure pool. Canonical:
> PART4 §17 (rev 2, 2026-09-19 — the authoritative model) + memory
> `project_sub_structure_pool_architecture.md` (rationale history).
>
> **⚠ SUPERSEDED by Plan C (2026-09-20).** The next four entries describe the
> 2026-07-08 rev-1 model (landed as Stages 1–3.2a, found wrong at 3.2b,
> reverted). Read them as history of what was tried and why it was wrong. The
> rev-2 rules that replace them, as landed in `multitf/sub_structure_pool.py`,
> `multitf/lifecycle_sweep.py`, `multitf/parent_tables.py` and
> `multitf/entity_df_mutation.py`:
>
> - **Identity — unchanged.** A unique sub (`PooledStructure`) is
>   `StructureKey(parent_path, sub_tf, direction, starting_idx)` with ABSOLUTE
>   `direction`; `sub_id` is global creation order (§17.3). A `TriggerRecord`
>   is `(lens, parent_sid, parent_cycle_id, trigger_sub_sid)` with FK `sub_id`
>   (never None); `trigger_sub_sid` starts at 0 per `(lens, parent_sid,
>   parent_cycle_id)` and is CREATION-ordered (§17.4). `sub_sid` no longer
>   exists on any structural artifact (only main `SidRecord`s keep
>   `sub_sid = structure_id`); the chart identity of a sub is `sub_id`.
> - **`relative_dir` vs `lens` (§17.3).** Two record fields, neither on the
>   structure: `relative_dir = "confluence"` iff `direction ==
>   parent_sd[parent_sid]` (`CTS_ESTABLISHED.meta["struct_direction"]`) else
>   `"counter"` — semantic; `lens` = the chart the record draws on:
>   `resolve_lens(use_case)` for the four named types, and the SPAWNING
>   record's lens for a `reversal` record (sticky-per-chart). They legitimately
>   differ (sub `2639/−1` is `relative_dir=counter` on the confluence chart).
>   A sub's `relative_dir_segments` is the step function over its records.
> - **Replacement is scoped per `(lens, parent_sid, parent_cycle_id, sub_tf,
>   direction)` — SAME-LENS ONLY.** A new record of that scope mapping to a
>   DIFFERENT sub ends the incumbent record with `end_reason =
>   "same_dir_replacement"` at the new record's true `start_idx` (not its
>   `trigger_idx`, not its finalize). A counter record never ends a
>   confluence record; the rev-1 pool-wide "≤1 active per direction" rule and
>   its cross-chain reversal interaction are RETRACTED. Invariant: ≤1 active
>   record per scope (`SubStructurePool.active_record` asserts).
> - **Per-record lifecycles, aggregated into the unique sub (§17.4–§17.5).**
>   Record: `start_idx = max(probe_finalize_idx, trigger_idx,
>   parent_floor_idx)`; `trigger_end_idx` = the first of {own reversal,
>   same_dir_replacement, parent_end} (equal-idx priority `reversal >
>   parent_end > same_dir_replacement`, `_END_REASON_PRIORITY`); `end_idx =
>   max(trigger_end_idx, start_idx)`; `is_zero_length = trigger_end_idx <=
>   start_idx` → participates in nothing. Sub: `start_idx` = the first
>   non-zero-length record's `start_idx`; `end_idx` = over the live records
>   that EXIST at `t` (`start_idx <= t`), `min` of the record `end_idx`s
>   STRICTLY `> max_start` — start-before-end at the same idx is what keeps
>   a sub continuous across a same-candle handover (see "The Sweep Phase
>   Order Is Load-Bearing" below).
> - **A RECORD cannot outlive its parent structure; a unique SUB can.** The
>   parent bound is the record's `parent_end` (`ParentTables.end(S, C)` =
>   LOH of the next cycle's clamped start, else the sid's reversal, else
>   None); the sub inherits it only through the aggregation rule, so a sub
>   spans parent cycles AND parent sids (sub `2365/+1` runs `[2470, 2829]`
>   across H1 (0,0)→(0,1)).
> - **Run cap = the DATA EDGE** for every sub (`build_or_get_geometry`,
>   `run_cap_abs = len(m15) - 1`) — a compute bound only; the lifecycle
>   projection (`render_sub_projection`) applies the sub's `[start_idx,
>   end_idx]` once and mirrors it into every lens df in `sub.lenses()`. The
>   knowable-at clip on render (`knowable_at_idx`) keys `BOS_CONFIRMED`,
>   `CTS_ESTABLISHED` and pattern-path `CTS_UPDATED` on their moment
>   `confirmed_at` (the last two since Plan E E3b) and everything else on
>   `ev.idx`, so only `REVERSAL_CANDIDATE` can still straddle a cap (known limit,
>   PART4 §17.12; see "Sub-Structure Pool: Run Cap ≠ Lifecycle End…" below).
> - **Probe cache** keys `(parent_path, sub_tf, direction, initial_input_idx)`
>   (unchanged in shape; first-probe-is-truth, `ProbeCacheEntry`). See "Probe
>   Cache Keys Are Shared by Reversal Handoffs" for the measured hits.
>
> Measured on the first Plan C replay (2026-09-20, reference window): 8 subs /
> 11 records / 4 unresolved (all `degenerate_parent_cycle`), every sub window
> and record start/end/reason equal to the predicted table in
> `memory/reference_pool_redesign_groundtruth.md`; `sub_id`s 0–7 in creation
> order (454/+1, 1797/−1, 2365/+1, 2639/−1, 3304/−1, 3621/+1, 3760/−1,
> 4027/+1); H1 8 of 9 CSVs byte-identical (`_wvmi.csv` header-only: the
> shared exporter column `sub_sid` → `sub_id`).

**Rule:** Under the pool, a sub structure's identity is `(parent_path, sub_TF,
direction, starting_idx)` with `direction` **absolute (+1/−1)**. Confluence vs
counter is a **derived, per-chart label** (`direction == parent.current_sd`), NOT
a sticky entity property and NOT part of identity.

**What changed from Phase 1:** §2 said `starting_alignment` is "sticky for the
structure's lifetime — does not flip on internal reversals," and the code filed a
reversal-born sid under its spawning entity (confluence stayed confluence). Under
the pool a reversal ENDS the sub and spawns a NEW opposite-direction sub, which
is classified by ITS OWN direction. So:

- **Cross-chain reversal interaction (new, easy to miss):** a confluence-direction
  sub (dir == parent_sd) that reverses spawns a dir == −parent_sd sub, which is a
  **counter** sub. Per "at most 1 active per `(parent_path, sub_TF, direction)`",
  that new counter-direction sub **ends any active counter-direction sub** — a
  cross-chain end the Phase-1 sticky-entity code does NOT have (there the
  reversal-born sid stayed in the confluence entity, untouching counter). Don't
  reintroduce sticky-entity classification; it breaks the ≤1-active-per-direction
  invariant.
- **Sticky survives ONLY as chart attribution:** a `reversal` trigger attributes
  its sub to the **same chart** as the sub it reversed from (so a reversed
  confluence structure still appears on the confluence chart), PLUS any chart
  whose own trigger reaches the sub. So a single unique sub can render on BOTH
  charts. This is chart membership, not structure identity — don't conflate them.

---

## Sub-Structure Pool: Lifecycle End Is `min(end ≥ max start)`, NOT Global-Earliest (Phase 2)

> *(rev-1 entry — HISTORY; superseded by Plan C 2026-09-20, see the banner under "Sub-Structure Pool: Alignment Is Direction-Derived…" above.)*

**Rule:** A unique sub has ONE continuous lifecycle: `start = min(trigger_dt)`
over all mapped triggers; `end = min( end-candidate : end-candidate ≥
max(trigger_dt) )` (else `None`/open). End candidates = `{own reversal,
same-direction replacement start, parent-lifecycle-end}`.

**Why not global-earliest end (the trap):** a cross-parent-cycle sub (canonical:
M15 3304/−1, triggered in parent cycle 1 AND cycle 2) has an end candidate at the
parent-cycle-1 boundary that sits BEFORE its cycle-2 trigger. Global-earliest
would force it to end at that boundary and restart — a degenerate same-dt
start/end, splitting one real structure into two. Excluding end candidates
earlier than the LAST trigger keeps it continuous across the intervening
boundary. `start` uses earliest trigger; `end` uses first genuine end after the
latest trigger.

**Edge to watch (not fixed in v1):** if a same-direction sub genuinely replaces
this one and THEN a later trigger re-resolves to this sub's exact `(dir, start)`
(possible — the probe is deterministic), `min(end ≥ max start)` over-extends the
sub across the gap where it was actually replaced. Symptom: overlapping
structures on one chart in `/compare` review. Precise fix if it bites: interval
sweep instead of a single window. Start with the single-window rule.

**Start/end feed the clamps.** These are the values passed to
`compute_cycle_lifecycle` floor/cap — projected ONCE per unique sub in a post-pass
(geometry is derived once, cap-free, at creation), NOT once per trigger record.

---

## Sub-Structure Pool: Dedup Reuse Is Byte-Identical ONLY Given a Deterministic Natural-End Build

> *(rev-1 entry — HISTORY; superseded by Plan C 2026-09-20, see the banner under "Sub-Structure Pool: Alignment Is Direction-Derived…" above.)*

**Rule:** Reusing a `PooledStructure` for a second trigger (instead of rebuilding)
is byte-identical ONLY because (a) the probe is deterministic given
`(parent_path, sub_TF, direction, initial-start)` and (b) the structure runs to
its **natural end** (first reversal), so its slice/run is independent of which
trigger asked. If either assumption breaks, dedup silently returns a structure
that differs from an independent rebuild.

- **Do NOT key the pool including `end_idx` / trigger boundary.** The whole point
  is that two triggers with the same `(dir, start)` but different `end_idx`
  resolve to the SAME final start and SAME structure. Keying on `end` yields zero
  dedup (the real dupes have different ends).
- **Probe cache** keys on the INITIAL start `(parent_path, sub_TF, direction,
  initial-start)` and accepts that different `end_idx` return the same final
  start (explicit approximation). Structure pool keys on the FINAL start.
- **Validate dedup by chart-visual + event-count parity**, not per-row M15 CSV
  (storage representation changes — two entity dfs → one store — so per-row M15
  parity will not hold, per §13). Expected intended deltas: duplicate-structure
  collapse on charts, `sub_id` renumber, natural-end window-end effects,
  knowable-at clip. Anything else is a regression.

---

## Sub-Structure Pool: Run Cap ≠ Lifecycle End; Knowable-At Clip on Render

> *(rev-1 entry — HISTORY; superseded by Plan C 2026-09-20, see the banner under "Sub-Structure Pool: Alignment Is Direction-Derived…" above.)*

**Rule:** Keep three bounds distinct — they are easy to conflate:

| Bound | What | Value |
|---|---|---|
| **Run cap** | how far MS is allowed to compute (slice upper) | `min(parent-structure-end, data-edge)` |
| **Natural end** | the structure's first reversal (data-intrinsic) | found by the run; ≤ run cap |
| **Lifecycle end** | when the sub stops being current in a lens | `min(end-candidate ≥ max start)` (§ above) |

The run cap must be ≥ the natural reversal (else the reversal is missed) but is
otherwise a pure compute/cost bound. **On render into a lens, events are clipped
by `knowable_at_idx`** (`multitf/sub_structure_pool.py`, applied by
`pooled_structure_build.clip_events_to_window`) — NOT by `ev.idx` uniformly.
What the code does: `BOS_CONFIRMED`, `CTS_ESTABLISHED` and pattern-path
`CTS_UPDATED` are keyed on their moment `meta["confirmed_at"]` (the last two since
Plan E E3b, 2026-09-25), every other type on `ev.idx`. The `BOS_CONFIRMED` case
shows the point: clipping it by
its anchor (`ev.idx` until Plan E E4b) would surface a BOS whose anchor is inside the
window but whose confirmation landed past it — an event the Phase-1 bounded run
could not have known.

`ev.idx` is the knowable-at candle only for events whose `ev.idx` IS their
moment (`CTS_CONFIRMED` / `CTS_RECONFIRMED` — `ev.idx == confirmed_at` —
raw-path `CTS_UPDATED`, `STATE_CHANGED`, and `CTS_ESTABLISHED` / `BOS_CONFIRMED` /
pattern-path `CTS_UPDATED` since Plan E E4a / E4b / E4c — the clip keys the last
three on `confirmed_at` (the pattern path since Plan E E3b), from the time their
`ev.idx` was the retro-stamped anchor). It is NOT for
`REVERSAL_CANDIDATE` (applies at `meta["apply_idx"]`), which can still straddle a
cap (`ev.idx <= cap <` its apply) and survive the clip, yielding a half-derived
reversal — the remaining known limit in PART4 §17.12 (zero straddles on the
reference window). Since 2026-09-27 the fib terminal and the prev-BOS line read the
realised `STATE_CHANGED` (clipped at the cap), so a straddling candidate no longer
stamps a fib 'reversal' past the cap; only the chart's candidate markers still read
the candidate (its anchor / pattern, never the apply).
See ARCHITECTURE "`ev.idx` convention".
The rule for any new or changed clip: key each type on its moment column, never
on `ev.idx` by default. Changing this clip is its own `/compare`.

---

## The Sweep Phase Order Is Load-Bearing (Plan C, 2026-09-20)

**Where:** `multitf/lifecycle_sweep.py` (`_Sweep.run` + `_phase0_fire` /
`_phase0_spawn` / `_resolve_and_record` / `_phase1_record_start` /
`_phase2_sub_start` / `_phase3_record_end` / `_phase4_sub_end`), PART4 §17.6.
The sweep is a priority-queue over moments `(idx, phase, order_key, push_n)`;
every rule in §17.4–§17.5 is monotone and write-once, which is the only reason
the sweep is equivalent to iterating candle-by-candle. The ordering rules below
are not implementation details — each one changes a lifecycle value if broken.

**1. Phases 0→4 at each idx, each to completion before the next.**
`0 TRIGGER_FIRE / REVERSAL_SPAWN` (resolve → create record → queue its
`RECORD_START` and any end already known) → `1 RECORD_START` → `2 SUB_START` →
`3 RECORD_END` → `4 SUB_END`. Within phase 0, `REVERSAL_SPAWN` sorts before
`TRIGGER_FIRE` (`_KIND_REVERSAL_SPAWN = 0 < _KIND_TRIGGER_FIRE = 1`) and
`TRIGGER_FIRE`s order by `(lens_rank, type_rank, trigger_event_idx)` —
confluence before counter, `first_*` before `subsequent_*` (the second
trigger's sibling read at `hi = t` inclusive can see the first's sub). Every
other moment orders by record `seq`.

**2. Start-before-end at the same idx is what keeps a sub continuous across a
same-candle handover.** Phase 4's candidate filter is STRICT: `end_idx >
max_start` over the records that EXIST at `t`. When record A ends at `t` and
record B (a different record of the same sub) starts at `t`, phase 1 starts B
first, so at phase 4 `max_start = t` and A's end `t > t` is false → the sub
persists. Reference-window cases (both §9.2 fixtures): sub `2365/+1` — its
(0,0) record ends `parent_end` at 2611 and its (0,1) record starts at 2611 →
one sub `[2470, 2829]` ending on its own reversal (rev 1's `>=` rule would
have ended it at 2611 and restarted it — §17.5); and the `3304/−1` → `3760/−1` handover at 3819 on
the confluence lens — sub `3760/−1`'s record starts at 3819 in phase 1, which
queues the `same_dir_replacement` end on `3304/−1`'s record for phase 3 of the
SAME idx; `3304/−1` then ends at 3819 (`3819 > max_start 3621`) and the two
live windows share the boundary candle as half-open intervals `[3621, 3819)`
/ `[3819, …)` — no gap, no overlap. Both depend on phase 1 running before
phase 3 at one idx.

**3. The heap is idx-monotone — never queue a moment in the past.** `_push`
asserts `(idx, phase) >= (cur_idx, cur_phase)`; `run` asserts each pop's idx
≥ the previous. Consequence at build time: a sub whose `natural_reversal_idx
R <= idx` of the trigger that built it spawns NOTHING (no record can be live
at `R`, every `start_idx >= idx > R`); step 6 makes that record zero-length
instead. A `REVERSAL_SPAWN` is queued only when `R > idx`, and only once, when
the geometry is CREATED (`created is True`).

**4. A record EXISTS only from its `start_idx`.** The object is instantiated
at phase 0 of its `trigger_idx` (every historical field is known then), but
until its `RECORD_START` moment it is invisible to every lifecycle
computation: not in `max_start`, not an end candidate, not an incumbent, not
a lens member, not a sibling-read candidate (`start_idx <= hi`). An FC record
created at 463 that starts at 1020 must not end anything, be ended by
anything, or be seen by anything in between. "All the existing records" in
the §17.5 sub-end rule means exactly the STARTED ones (`start_idx <= t`).

**5. Zero-length records participate in nothing.** `is_zero_length =
trigger_end_idx is not None and trigger_end_idx <= start_idx` — defined on
`trigger_end_idx`, NOT `end_idx`, so it is already correct in phases 1–2 of
the idx at which the record is ended (phase 3 writes `end_idx` later the same
idx). Not the sub's start, not `max_start`, not an end candidate, not an
incumbent, not lens membership (`live_records()` / `lenses()`), not rendering,
not a sibling-read candidate. Logged only (`_triggers.csv`, `is_zero_length`
column). Two sources: a post-end re-trigger (step 6: `sub.end_idx is not None
and rec.start_idx > sub.end_idx` → frozen with the sub's end, linked via
`ended_by_sub_id`) and a sub whose own reversal / parent end precedes the
record's floored start (step 6: `e[0] <= rec.start_idx`). A record that
reaches phase 1 after its sub froze (`sub.end_idx < rec.start_idx`, strict —
`==` is the handover of rule 2) is frozen there.

**6. The phase-1 incumbent is a STARTED record.** The incumbent test filters
to records whose `RECORD_START` has run (`r.seq in self._active`) AND
`r.is_active_at(t)` — not `pool.active_record` alone. The interval rule alone
would also see a same-idx record whose start moment is still queued (a later
`seq`), inverting the "later `seq` replaces earlier" collision rule; and a
same-idx collision's incumbent is frozen zero-length IMMEDIATELY in phase 1
(never queued to phase 3) so phase 2 cannot read it as live. GOTCHAS
"Phase-1 Incumbents Must Be STARTED Records, Not Interval-Active Ones".

**7. `RECORD_END` entries for an already-ended record are stale and dropped.**
Every entry carries `(idx, reason, ended_by)`; phase 3 applies the
highest-priority entry at `t` (`reversal > parent_end >
same_dir_replacement`) only if `rec.trigger_end_idx is None`. Earliest idx
always wins across idxs because earlier entries run first and later ones
become stale; a stale entry does not mark the sub "touched" and does not
queue a `SUB_END`.

**Post-sweep asserts (`_after_sweep`)** — if any fires, the order was broken:
every live record has `start_idx < end_idx` (or open) and `start_idx <=
sub.end_idx`; every sub with a `start_idx` has `end_idx > start_idx` (or
open); a sub without `start_idx` has no live record; per `(lens, parent_sid,
parent_cycle_id, direction)` the live windows are non-overlapping as
HALF-OPEN intervals `[start_idx, end_idx)` (a handover shares its boundary
candle: `[3621, 3819]` / `[3819, …]`).

**Testing this:** the predicted-table test
(`tests/test_lifecycle_sweep_predicted_table.py`, stubbed probe + geometry,
< 1 s) cannot distinguish rules 2/4/6 from a wrong ordering (no sub on the
reference window has a pending record across an end); the unit tests in
`tests/test_lifecycle_sweep_unit.py` must. Add a unit case, not a replay, when
touching the phases.

---

## Bounded MS Runs Must Not Read Past `end_idx` (FIXED — Plan A, 2026-09-19)

**Rule (the definition):** a bounded run `MarketStructure(df, start_idx=s,
end_idx=B).run()` produces exactly the events and output rows of
`MarketStructure(df_B, start_idx=s, end_idx=None).run()`, where `df_B` is the
same frame **truncated to `[0, B]`** (look-ahead feature labels recomputed on
the truncated frame). `effective_end = min(n-1, B)` **is the run's data
edge**: nothing past it is read, and everything the run does at the edge is
what it already does at the real data edge. In live those are the only
candles that exist. Enforced by `MarketStructure.__init__` (`_effective_end`)
and checked by a **post-run assert** (`max(ev.idx) <= effective_end`, right
after the main loop) and by a property test over EVERY bound of every fixture
(`tests/test_ms_bounded_equals_truncated.py`), so it cannot pass by fixture
luck. **Do not weaken the assert** — if it fires, a forward read was missed.

**Why it matters:** a bounded run must be reproducible from its stated bound.
Before the fix, FC(1,0)'s leaked `CTS_CONFIRMED@2844` (run bounded at 2843)
was the *only* reason its `finalize_idx` was 2844 and its retrace window
`[2761..2841]` instead of `[2761..2843]` — a leak can move `starting_idx`,
i.e. the pool key. `unified_probe`'s "inclusive supreme upper bound" and
`compute_bounded_structure`'s "never writes/emits past it" were false.

**The five leak sites (all in `market_structure.py` unless noted; all fixed
by clamping to `self._effective_end` instead of `len(self.df) - 1`):**

| # | Site | Leak | Fix |
|---|---|---|---|
| L1 | `_step_anchor` / `_post_apply_range_check`: `D = min(i + range_max_k, n-1)` | a pattern anchored at `i <= B` applied at `i+k > B`; the range back-fill stepped candles `> B` (per-candle CTS updates, proximity, BOS probes) | `D = min(i + range_max_k, effective_end)` |
| L2 | `_finalize_range_candidate_offline` stamps `RANGE_STARTED` / `STATE_CHANGED→RANGE` at the **pre-computed `is_range_confirm_idx` label** (`range_label.py`, full-frame, locks the FIRST close in `[i+2, i+5]` inside candle `i`'s range) — never compared to the bound. **Clamping `D` alone does NOT touch this path** (the FC(1,1) `@3050` leak on a bound of 3047) | `_is_range_candle_given_confirm`: `confirm_idx > effective_end → (False, None)` — equivalent to recomputing the label on the truncated frame |
| L3 | `patterns/structure_patterns.py`: the six `len(df)` guards in the detectors + confirmation helpers; once L1 is fixed `detect_best_for_anchor`'s priority rule can still let a future candle pre-empt a knowable 2-candle pattern (`continuous` SUCCESS at `idx+2` wins over a 2-candle SUCCESS at `idx+1`; with `B = idx+1` the full frame returns the continuous → dropped → NO pattern, the truncated frame returns the 2-candle → applied) | `BreakoutPatterns(df, end_idx=…)` → `n_visible`; MS and `unified_probe` Phase 1 pass their bound (`pattern_engine.py`'s offline `pat*` pass stays unbounded — MS never reads `pat*`) |
| L4 | `_start_reversal_watch`: `expires_idx = min(i+5, n-1)`; `_maybe_expire_reversal_watch` / `_rewind_to`: rewind targets clamped to `n-1`. **How the watch works:** a watch survives its anchor only if `_schedule_reversal_from_anchor` found a reversal pattern with `apply <= expires_idx` (else `rv_anchor_failed` clears it at once); (until F3b, 2026-09-29) expiry fired only when the pending apply equalled `expires_idx`, because `_maybe_expire_reversal_watch` ran **before** `_maybe_apply_pending_reversal` in `_replay_step_no_patterns` (both the order and the expiry are gone since). So with `expires_idx` at `n-1` a reversal applying in `(B, i+5]` was scheduled (run ends with an open watch + pending reversal, `expires_idx` meta past the bound) whereas a frame ending at `B` never schedules it | `expires_idx = min(i+5, effective_end)`; rewind targets clamp to `effective_end` (the rewind was removed 2026-09-29, F3b). Consequences (decided, = today's data edge): apply `> effective_end` → never scheduled; apply `< effective_end` → reverses as before; apply **exactly** `effective_end` → **reverses at the edge on every path** since F3b (2026-09-29; MARKET_STRUCTURE_SPEC "A reversal confirming on E applies": the pending apply precedes the expiry). Until then it was discarded as a false break (`probe_no_break` at the edge, rewind to `anchor+1`) on the pending-apply path only — the winner path applied it — and that edge false break was a repaint: one more candle moved the expiry past the edge and the same pending applied AT the old edge (112 of 141 measured edge discards). Identical in bounded and truncated runs either way |
| L5 | the two resolvers MS hands `self.df` — `_bos_inner_resolver` at `BOS_CONFIRMED` (`compute_bos_inner_from_event`) and `_poi_inners_resolver` at `CTS_ESTABLISHED`/`CTS_UPDATED` — derive base patterns whose reads (`identify_base_pattern` inside-bar scan to `anchor+5`, `find_base_threshold` to `i+5`, 2-candle/star `+1`, `zone_thresholds` `+1/+2`, `kl_zones_v1.py`) are clamped to the **frame**, not the bound; a BOS inner derived from candles `> B` feeds `_maybe_confirm_cts_via_proximity` at candles `<= B` | `_resolver_df()`: the resolvers get `self.df.iloc[:effective_end+1]` (RangeIndex, `loc == iloc`, attrs propagated), built ONCE per run and cached — an `iloc` slice deep-copies `attrs` (see GOTCHAS "Per-cell `.iloc[]`…"), and on the probe frame (which carries the mirrored sub attrs) paying that per `CTS_UPDATED` is far too slow. Main / unbounded runs and frames that already end at the bound get `self.df` (fast path, no change). The resolvers never write to the frame — keep it that way (a write through the view would land in `self.df` silently) |

**Who was actually affected:** every sub geometry build already slices its
frame to the bound (`_build_or_get_sub_geometry` then — since Plan C
`entity_df_mutation.build_or_get_geometry` → `_build_geometry`: `iloc[slice_begin:run_cap+1]`,
`compute_imbalance` + `is_range_*` re-derived on the slice, `end_idx =
run_cap_in_slice`), so `n-1 == effective_end` there and the fix is a no-op by
construction; main runs pass `end_idx=None`. The only production path that ran
MS on a frame longer than its bound was **`unified_probe._run_phase2`** (the
first_confluence probe: `df_probe = df.copy()` of the whole M15 entity frame,
`end_idx = probe_end_idx`). Measured with `debug/probe_fc_finalize.py` on the
2025-11→2026-01 window: leaks FC(0,0) +4 (`RANGE_STARTED`/`STATE_CHANGED@1725`,
bound 1721), FC(1,0) +1 (`RANGE_STARTED`/`CTS_CONFIRMED`/`STATE_CHANGED@2844`,
bound 2843), FC(1,1) +3 (`@3050`, bound 3047) → all 0 after the fix; the only
probe-row change is FC(1,0) `finalize_idx` 2844 → 2843 (`no_retrace`,
else-branch), `starting_idx` 2803 unchanged. Phase 1 (`find_true_first_breakout`)
was already equivalent (`est > hi` drop) — H1 byte-identical.

**Known residual (documented, not fixed — narrowed by Plan F, 2026-09-24):**
`df.attrs["imbalances"]` is a full-frame instance list — an FVG whose `c2 == B`
exists only because `c3 = B+1` was seen, and a merged run ending at `B-1` gets
its bounds from `B+1` (`imbalance.py`). The MS consumers — `_update_cycle0_data`
and `_refresh_poi_inners_for_cycle` → `compute_poi_inners_for_cycle`
(`find_ic_candidates` IC cond3; `select_fib_anchor_for_cycle` cond1/cond3) — ask
with `evaluated_at=None`: as-of for *fills* (`check_to_idx <=` the refresh moment
`<= B` — `st.cts.idx`, or since Plan E E3a the triggering event's moment for
Scenario-2 cond1; the apply / processing candle is never past the edge),
no existence cut, by decision (MARKET_STRUCTURE_SPEC "Snapshot vs per-candle").
What each half can reach in a bounded run's events / rows:
- **existence — nothing.** Every window ends at or before `st.cts.idx`, so a gap
  whose c3 is past `B` is counted only when its c2 `== B == st.cts.idx`; the
  snapshot is read at `i > st.cts.idx` and `cycle0_data` at a later cycle-1
  refresh — candles the run never processes.
- **merged bounds — only a degeneracy mismatch.** The truncated frame's
  instance is the full run's formed prefix at `B`; asked with `check_to_idx <=
  B`, both are unfilled (the c3 rule's exactness) unless the prefix and the
  merged run differ in degeneracy (`gap_size <= 0` counts as filled) — 0 cases
  on the reference data (IMBALANCE_FILL_SEMANTICS.md "Knowability — the c3
  rule").

Sub builds are immune (they re-run `compute_imbalance` on the slice). If the
property test ever trips on it, record and decide — do not silently exclude it.

**What this is NOT — prefix (clip) equivalence.** "Clip of the natural-end run
≡ bounded run" does NOT hold in the last `range_max_k` (5) candles before `B`,
inherently: the natural run may be back-filling a pattern that applies past
`B` while the bounded run processes those candles as anchors; a reversal
pending past `B` is scheduled in one and not the other (a pending reversal
applying **exactly at `B`** reverses in both since F3b, 2026-09-29 — before, it was
a false break in the bounded run but a reversal in the natural run, and an extra
rewind rebuilt from 0). This is the 5-candle
pending-confirmation nature of MS, not a bug — it is why the pool (Plan C)
runs geometry to the data edge and clips *windows* out of one run. Do not try
to make prefix-equivalence hold. **"Look ahead but suppress emission" is NOT an
option either** — it keeps future information inside the state machine.
`test_pooled_structure_build`'s prefix-family tests pass on their fixture
(cap = R−6); they are fixture-dependent by nature.

**Deliberately OUT (L5b):** the *probe's* ad-hoc BOS_0 derivation at a reset
candidate (`unified_probe._bos0_inner_at_start` →
`build_ad_hoc_bos0_reference_zone` → `identify_base_pattern`) reads up to
`candidate+5` past `probe_end_idx`. A search-bound reproducibility gap, not a
live-causality one (those candles exist at trigger time), and closing it can
move the H1 sid-1 start through the main reversal probe. Needs its own
decision + `/compare` — see "Probe `end_idx` Is the Supreme Bound".

**Shared 5-candle horizon:** `range_max_k` (5) = the detector's max
confirmation offset (`idx+5`) = `RangeLabelConfig.max_lookahead` (5) = the
inside-bar scan half-width (5). One horizon; do not decouple them.

Plan: `plans/PLAN_A_ms_bounds_leak.md` (audit table of every forward read,
classified). Reproduce / re-measure with `debug/probe_fc_finalize.py`.

---

## Degenerate Parent Cycles: Log the Trigger, Don't Build the Sub (2026-09-19)

**Rule:** A parent cycle whose lifecycle **floor ≥ lifecycle end** (zero or
negative length) is *degenerate*. Every sub trigger inside it is provably
inert — its record's `start_idx = max(finalize, floor) ≥ floor ≥ end ≥
trigger_end`, so `end_idx == start_idx` no matter what the probe returns. For
such triggers: **write an unresolved-trigger log row (`reason =
degenerate_parent_cycle`, no `sub_id`), and do NOT run the probe, MS, or
downstream.**

**Why they exist:** the reversal handoff makes a post-reversal parent's
lifecycle start at the prior sid's reversal (`compute_struct_start_by_sid`),
but that parent's early cycles were built retroactively *before* it. H1 sid 1
on the 2025-11→2026-01 window: `struct_start = 902`; cycle (1,0) spans H1
703→748 (M15 floor 3611 > end 2995 — inverted), (1,1) spans 748→902 (floor
3611 == end 3611 — zero). Only (1,2) is a real cycle. Every sub anomaly that
motivated the pool redesign traces to those two cycles.

**Why not probe anyway (to link the record to a unique sub):** three of the
five trigger types resolve their probe input from the *sibling's* CTS events;
if the sibling in a degenerate cycle was never built, the probe falls back to
the ad-hoc BOS_0 and returns a **different `starting_idx`** — a phantom pool
key that never gets geometry. Post-end re-triggers in a LIVE cycle are the
opposite case: probe normally, link to the frozen sub, skip MS (zero-length
`TriggerRecord`, has `sub_id`, participates in nothing but the log).

**Consequences to accept:** memberships from degenerate cycles vanish from live
subs (e.g. sub `3304/-1` loses its (1,1) records and its counter-lens
presence); reversal-born successors inside a degenerate cycle never fire (the
reversal is equally inert). **Log a WARNING per degenerate parent cycle** — it
is the clearest signal of a retroactive parent, and nothing surfaces it today.
Under 3.2a these subs rendered as outline-only `status="inactive"` zones; under
3.2b (`9fd3143`, superseded) they were wrongly LIVE because the parent floor was
dropped on the render path.

**Implemented by Plan C (2026-09-20).** `multitf/parent_tables.build_parent_tables`
computes `degenerate[(S,C)] = end_m15 is not None and floor_m15 >= end_m15`
(with `end_h1[(S,C)] = floor_h1[(S,C+1)]`, the next cycle's CLAMPED start —
so (1,0)'s end is 3611, not the raw 2995; degenerate either way) and logs one
`WARNING [parent_tables] degenerate parent cycle (S,C): floor=… end=…` per
cycle. `lifecycle_sweep._Sweep._resolve_and_record` step 1 checks
`tables.is_degenerate(S, C)` BEFORE the probe and appends
`UnresolvedTrigger(reason="degenerate_parent_cycle", detail="floor F >= end
E")` to `pool.unresolved` (`[sweep] UNRESOLVED (skipping) …`; exported to
`*_M15_unresolved_triggers.csv`). A reversal-born spawn inside a degenerate
cycle takes the same path (it is resolved through `_resolve_and_record` too),
so it is inert as specified. Measured on the first Plan C replay (reference
window): exactly the four (1,0) / (1,1) triggers are unresolved — FC(1,0),
FC(1,1), `first_counter`(1,1), `subsequent_confluence`(1,1) — and no other
reason appears; the (1,1) counter reversal at 3589 never fires because
`3230/+1` is never built. "Memberships" above = rev-1 vocabulary for what are
now `TriggerRecord`s.

---

## Probe Cache Keys Are Shared by Reversal Handoffs (Plan C, 2026-09-20)

**Rule (PART4 §17.8, as landed):** the probe cache key is
`(parent_path, sub_tf, direction, initial_input_idx)` with `initial_input_idx`
ENTITY-ABSOLUTE for EVERY probe — the FC probe's price-mapped BOS anchor
(`_probe_with_cache`, `multitf/entity_df_mutation.py`), a sibling type's
`ref_zone.anchor_idx` (same wrapper), AND the reversal handoff's input
(`_resolve_reversal_start`: the probe runs slice-local on the reversing sub's
`bounded.df`, but the cache is read/written with `input_abs = probe_input_local +
slice_begin` and the cached `starting_idx` / `finalize_idx` are shifted the same
way). "Same probe" = same direction + same initial input; the FIRST probe to
finalize for a key is the truth for every later probe of that key, regardless
of its `probe_end_idx` and regardless of its reference zone (accepted
approximation, user decision 2026-09-19). A hit returns the cached
`ProbeCacheEntry` (`starting_idx, finalize_idx, finalize_condition, bos0_inner,
probe_end_idx`) and SKIPS `unified_probe`; `[probe_cache] hit` when the bound
equals the cached one, `[probe_cache] APPROX hit` otherwise, `[probe_cache]
REF-ZONE DIFFERS …` when the hitting trigger's reference inner is not
`isclose` to the cached probe's OWN reference inner (`ProbeCacheEntry.ref_inner`
— iteration 1's threshold). NOT the cached `bos0_inner`: that is the FINAL
iteration's threshold, which moves on every reset (the first Plan C replay
compared against it and logged two spurious `REF-ZONE DIFFERS`; cold review
2026-09-20). A second write for a key with a different entry asserts
(`SubStructurePool.record_probe`).

**What was not foreseen:** the plan counted only H1-trigger pairs and
predicted ZERO hits on the reference window. But a reversal handoff probe and
an H1 trigger can resolve the SAME input candle in the same direction: the
handoff's input is the reversing sub's most recent qualifying CTS anchor
(`build_reference_zone_from_cts_event(...).anchor_idx`), and on this
window that candle coincided with the FC probe's price-mapped parent BOS
anchor (2365) and with two sibling reads' `anchor_idx` (2609, 4000).
Once the reversal key was made entity-absolute (it must be — the pool is
shared across subs), the two probes share the entry, and whichever ran first
in sweep order owns it.

**Consequences for the record (§17.4):** the later record INHERITS the cached
`probe_finalize_idx` (which may be EARLIER than its own `trigger_idx` — the
`trigger_idx` term of `start_idx = max(probe_finalize_idx, trigger_idx,
parent_floor_idx)` absorbs it) and its own Phase-2 probe is skipped, so there
is one fewer `[unified_probe phase2] early stop` line for an FC hit. `probe_finalize_condition`
is the cached probe's. The sweep's `finalize_idx == trigger_idx` assert for
non-FC types is bypassed on a hit (`ResolvedStart.cache_hit`). The record's
`parent_bos_anchor_idx` is still its own (the H1 BOS anchor that seeded an FC
probe; None for every other type — PLAN_E Q4).

**Measured on the first Plan C replay (reference window) — THREE hits:**

| hitting trigger | key (direction, input) | earlier probe that owned the key | inherited `starting_idx` / finalize | tripwire |
|---|---|---|---|---|
| FC(0,1), trigger 2611 | (+1, 2365) | sub-1 (`1797/−1`) reversal handoff at 2470 | 2365 / 2470 `no_retrace` (own Phase-2 probe skipped) | inners equal |
| `first_counter`(0,1), trigger 2843 | (−1, 2609) | sub-2 (`2365/+1`) reversal handoff at 2829 | 2639 / 2829 | inners equal (no tripwire) |
| sub-6 (`3760/−1`) reversal handoff at 4200 | (+1, 4000) | `subsequent_counter`(1,2), trigger 4083 | 4027 / 4083 | inners equal (no tripwire) |

No lifecycle value changed: each record's `start_idx = max(finalize,
trigger_idx, floor)` absorbed the inherited finalize (FC(0,1): `max(2470,
2611, 2611) = 2611`, previously `max(2608, 2611, 2611)`; `first_counter`(0,1):
`max(2829, 2843, 2611) = 2843`; the 4200 successor: `max(4083, 4200, 3611) =
4200`), and every inherited `starting_idx` equals the value the pre-pool
baseline had probed independently for that trigger (2365, 2639, 4027 — the
predicted table). The tripwire fired on none of the three once its comparand
was corrected (the two `REF-ZONE DIFFERS` lines of the very first replay
compared against the post-reset final `bos0_inner` — spurious). **Accepted
as designed (user decision, 2026-09-20)** — the
sharing is the specced key (PART4 §17.8: the reversal handoff's input is part
of the key space); the plan's "zero hits" was an enumeration error. Do not
"fix" it by widening the key (a reference-zone-in-key and a window-exact hit
rule were both considered and rejected on 2026-09-19) or by excluding the
reversal handoff (considered and not adopted 2026-09-20); either change is
its own `/compare`. Consequence to remember: a record's `probe_finalize_idx`
can be an INHERITED value earlier than its own `trigger_idx` — the
`trigger_idx` term of `start_idx` exists for exactly that.

**How to see it:** grep the replay log for `[probe_cache] hit` /
`APPROX hit` / `REF-ZONE DIFFERS`; the `_triggers.csv` rows show the inherited
`probe_finalize_idx` next to a later `trigger_idx`. The `/compare` skill's
`[unified_probe phase2] early stop` count drops by one per FC hit — that is
not a Plan B regression.
