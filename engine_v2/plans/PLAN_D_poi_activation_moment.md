# Plan D — POI activation gate on the cycle's established MOMENT (zones pass, 2026-09-22; amended 2026-09-23)

**LANDED 2026-09-23 as `0a4eadc`; save `20260923_172626_0a4eadc` (save commit `fba64f1`) — see §9.**

**Base:** `week8-volmom-multitf` after the docs-only `ev.idx`-convention + naming-standard commit (itself on `e2e0f89`).
**`/compare` baseline:** `artifacts/commits/week8-volmom-multitf/20260922_195430_aadb887` (24 CSVs, 3 charts).
**One cause:** the POI's cycle term becomes the cycle's established moment. That covers both the activation gate and
the exported meta field that records the gate's input. Nothing else moves in this commit.
**Evidence:** the zones-pass `/prepare` audit (9-agent workflow, adversarially verified) measured every number below
with independent live A/B harnesses, each reproducing the baseline's 24 CSVs byte-for-byte at HEAD. The variant
measured is "gate + meta re-valued" (harness variant B). The plan's cold review (3 lenses, each finding
adversarially verified) re-ran the §5 tests at HEAD and under the simulated fix. Session memory:
`project_zones_timing_audit_20260922.md`.
**Amendment 2026-09-23:** the user adopted the candle-index naming standard (GLOSSARY "Naming Standard"): a
past-participle name such as `*_established_idx` means a MOMENT. `cts_established_idx` therefore KEEPS its name and
now holds the moment, in both the code and the POI meta. This reverses the original decision 1 ("keep the meta key as
`CTS_ESTABLISHED.idx`") and drops the `cts_moment_idx` rename. §2–§6 and §8 are updated to match.

---

## 1. The defect

`zones/poi_zones.py` gates POI activation with:

```
first_active = max(cts_established_idx, ic_idx, lifecycle_floor_idx)      # _compute_poi_activation_history
cts_established_idx = CTS_ESTABLISHED(sid, cycle).idx                     # derive_poi_zones, :467-471
```

`CTS_ESTABLISHED.idx` is the cycle's CTS **anchor**: the breakout pattern's extreme candle, adopted as the CTS. It is a
historical price location, retro-stamped when the pattern applies. No CTS event is emitted when that candle closes, so
the cycle is not knowable there (ARCHITECTURE "`ev.idx` convention"). The cycle becomes knowable at its **moment**,
`meta["confirmed_at"]`. The moment is already:
- the canonical cycle lifecycle start (`structure_lifecycle.compute_cycle_lifecycle`, Plan C);
- the END of the previous cycle's POIs and KL zones.

The variable is named `cts_established_idx`, a moment under the naming standard, but it holds the anchor. Consequences:

- **Look-ahead.** A POI can activate `confirmed_at − idx` candles before its cycle exists (bound ≤ 5).
- **Double-live candle.** On such a cycle, two POIs are live over `[idx, confirmed_at)`: the previous cycle's (which ends
  at the moment) and the new cycle's (which starts at the anchor).
- **Collapsed-cycle contract broken** (no instance on this window). When the moment equals the cycle end, the POI still
  activates at the anchor and ends up `"ended"`. The contract requires `[]` / `"inactive"` (`structure_lifecycle.py`
  :90-93; POI_ZONES_SPEC §4, "If `first_active >= end_idx` the POI never activates").

The spec's own lead sentence already requires the fix: "POI activation is gated to start no earlier than the owning
cycle's lifecycle-start" (POI_ZONES_SPEC §4). Three places record this residual and defer it to "a zones pass":
- the "Open inconsistency" box under that lead sentence;
- the ARCHITECTURE "Activation floor" NB;
- the PART4 §17.6 "As landed" blockquote.

This is that pass.

## 2. Decisions (user, 2026-09-22 / 2026-09-23)

1. **`cts_established_idx` holds the moment**, `CTS_ESTABLISHED.meta["confirmed_at"]`, in the local, the
   `_compute_poi_activation_history` keyword and the exported POI meta key. The names are unchanged and the values are
   corrected. The anchor is not stored in POI meta; it stays available as `CTS_ESTABLISHED.idx` in the events CSV.
2. **Implementation:** read `CTS_ESTABLISHED.meta["confirmed_at"]` directly at the lookup, and keep the separate
   structure-start floor (`lifecycle_floor_idx = struct_start_by_sid[sid]`).
   - Considered and not chosen: reusing `cycle_life[(sid, cyc)][0]`, which is equivalent wherever a cycle-table entry
     exists.
3. **Fallback unchanged.** A `(sid, cycle)` with no `CTS_ESTABLISHED` keeps `int(fib_state.cts_idx)` as its cycle term.
   - That value is the fib's CTS anchor, not a moment. This is a KNOWN exception to the naming standard, documented at
     the site.
   - Its only live case (counter sub 5, cycle 1, IC 3654, value 3806) never activates, so the delta is zero.
   - That POI also lacks an end and is drawn to the chart edge. This is a queued follow-up (§7.3), where the exception
     gets resolved.

## 3. Code change (`zones/poi_zones.py` only)

**`derive_poi_zones`, the lookup (~:467-471):**

```python
cts_event = cts_established_by_key.get(key)
# The cycle's CTS-established MOMENT — the activation floor's cycle term (and the exported meta value).
# Fallback when the cycle has no CTS_ESTABLISHED: the fib's CTS anchor (a location, not a moment —
# a known naming-standard exception; such a cycle has no lifecycle entry; follow-up in PLAN_D §7.3).
cts_established_idx = (
    int(cts_event.meta["confirmed_at"]) if cts_event else int(fib_state.cts_idx)
)
```

- **The lookup indexes `meta["confirmed_at"]` directly**, with no `.get(..., ev.idx)` fallback. This is the guard of last
  resort. For every event that carries `structure_id` / `cycle_id` (the event contract), `compute_cycle_lifecycle` at
  ~:399 asserts `confirmed_at` first (`structure_lifecycle.py` :109-113). Test (g) pins the lookup's own guard with that
  upstream assert stubbed out.
- **Names don't change.** The keyword passed to `_compute_poi_activation_history`, its parameter, its body and the meta
  write (`"cts_established_idx": cts_established_idx`) keep their names. Only the value changes.

**Why the rest of the sweep needs no change** (verified live):
- Once `first_active` is past the anchor, the `CTS_ESTABLISHED` at the anchor moves from the in-window transitions into
  the pre-window loop (~:862-878). That loop applies `cts_idx` / `cts_price` before the sweep, so condition 1 and the
  variants at `first_active` are identical.
- Condition 3 needs an unfilled imbalance, and every such imbalance enters at `max(start, first_active)` (~:896). So a POI
  is always evaluated AT `first_active`; it cannot land later than the moment.
- In-window `CTS_UPDATED` stays on `ev.idx`. The raw path is causal. The pattern path records no moment, and fixing that
  is an event-contract change: parked for the convention plan, with one masked instance on this window.

**Comments corrected in the same edit** (naming-standard vocabulary):
- ~:366: the lookup feeds the gate and the meta field, not "confirmed_idx and end_time".
- ~:467: the lookup is the activation floor's cycle term, not "condition 1".
- ~:501-502 and ~:788-792: the floor formula gains the missing lifecycle-floor term.
- ~:862-866: "In practice ic_idx <= cts_extreme < cts_established_idx … this loop is a no-op" becomes
  "`CTS_ESTABLISHED.idx` (the CTS anchor) `<= cts_established_idx` (the moment) `<= first_active`". The pre-window loop
  applies the CTS state (the establishing event and any earlier `CTS_UPDATED`) whenever the anchor precedes
  `first_active`: on every cycle whose anchor precedes its moment, and on every POI whose IC or floor lies past the anchor.
  The IC CAN lie past the anchor, even past the moment, because IC candidates range to the fib's FINAL `cts_idx` (12/48
  POIs on the reference window). [Corrected by the landing cold review; the first wording claimed `ic_idx <= anchor`.]
- ~:187: the `scenario_context` docstring ("idx where CTS was established") is dead. Left alone (§7, Parked).
- The `POIZone` docstring ~:57-60 points at the POI_ZONES_SPEC field list instead of restating it.

**`zones/structure_lifecycle.py`** changes in its docstring only; it has no POI note. At :70-73, replace "(three M15 sub
cycles on the reference window; H1 equal only by luck)" with a pointer: "bound `confirmed_at <= anchor_idx +
range_max_k`; anchor == moment is the common case — ARCHITECTURE '`ev.idx` convention'". Lines :90-93 (the
collapsed-cycle contract) become true for POI with no text change.

## 4. Numeric prediction (stated before code; anything else = STOP and name the mechanism)

The fix changes 2 of the 37 activation calls' inputs (H1 5 + 32 unique sub POIs) and 1 output. The meta re-value changes
3 cells: the same 2 cycles, and sub 3 appears on both lenses.

| Artifact | Change |
|---|---|
| `_M15_confluence_poi_zones.csv` line 4 (sub 0 / sid 0 / cycle 2 / IC 678) | `confirmed_idx` 1223.0 → **1224.0**. In meta: `confirmed_idx` 1223 → 1224; `activation_history` `[{idx 1223, active, initial, [V30,V60,V90]}]` → `[{idx 1224, …same}]`; **`cts_established_idx` 1223 → 1224**. `status` "ended", `end_idx` 1721, `versions` ['V30'] and `current_versions` unchanged |
| `_M15_confluence_poi_zones.csv` line 19 (sub 3 / sid 0 / cycle 1 / IC 2808) | meta **`cts_established_idx` 2828 → 2829** only. Activation already 2829 (the sub's `start_idx` floor masks the gate change) |
| `_M15_counter_poi_zones.csv` line 4 (sub 3 / sid 0 / cycle 1 / IC 2808) | meta **`cts_established_idx` 2828 → 2829** only (same sub, mirrored) |
| the other POI rows | unchanged. This includes counter line 7 (the fallback POI, 3806) and all 5 H1 POIs (anchor == moment on every H1 cycle) |
| the other 22 CSVs | **byte-identical**: H1 ×9; M15 counter ×6 non-POI; M15 confluence ×6 non-POI; `_unresolved_triggers.csv` |
| H1 chart, M15 counter chart | identical at figure level. No chart reads `cts_established_idx` (verified by grep: its only readers are the env-gated debug print and the M15 mirror's shift list) |
| M15 confluence chart | **Figure diff = exactly 2 shapes + the customdata of 2 hover traces.** (1) The IC-678 POI's fill start (rect x0) moves one M15 candle, 2025-12-03 15:45 → 16:00. (2) Its confirm line moves the same way. The one-candle overlap with cycle 1's POI fill disappears; the two fills now meet at 16:00. (3) Its two invisible "M15 POI zone" hover traces show `confirmed_idx=1224` instead of 1223 (`customdata[4]`, via `export_m15_chart.py` :1584 → :1662 → :1683). **Counts unchanged:** H1 85/245, counter 153/125, confluence 294/233 |
| run.log | Fetch gate: `[data_bridge] Fetched 4228 M15 candles for NZD_USD in 5 chunks`, no `[data_bridge] ERROR`, the log newer than the raw CSVs, the `=== Replay Timing ===` block present. Silent-skip grep unchanged from the Plan C ground truth: exactly 4 `[sweep] UNRESOLVED` rows with `degenerate_parent_cycle` (= `_M15_unresolved_triggers.csv`), the two `WARNING [parent_tables] degenerate parent cycle` lines for (1,0)/(1,1), 3 `[probe_cache]` hits, no `REF-ZONE DIFFERS` |
| downstream consumers | none move. H1 `zone_proximity` (its POI gate goes through `poi_active_as_of`) runs for H1 main only, and H1 POIs are unchanged. H1 WVMI, sub WVMI, triggers, pool, KL, fib and events are all identical |

The count checks and `chart_census` cannot see the hover delta or the meta re-value. The hover delta shows in a
figure-level diff (step 4) and when hovering at chart review (1224 expected). The meta re-value shows only in the POI
CSVs' meta column.

## 5. Tests (tests-first; the fixture is the repo's live-MS candles with the moment after the anchor)

New file `tests/test_poi_activation_moment.py`. Fixture:
- `df = _prepare_df(_make_second_cts_moment_after_extreme_data())`, imported from `engine_v2.tests.test_unified_probe`
  (the established convention). The fixture's name keeps "extreme" until the naming migration renames it.
- `res = compute_bounded_structure(df, 0, +1)`.
- `out = _run_downstream_pipeline(res.df, res.events, +1, fib_mode="h1", skip_wvmi=True)` (the positional arguments are
  required). POIs are in `out["poi_zones"]`, the tracker in `out["fib_tracker"]`.
- Alternatively call `derive_poi_zones(res.df, res.events, fib_tracker=out["fib_tracker"], lifecycle_floor=…,
  lifecycle_cap=…, cap_reason=…)` directly, with a NON-None tracker. With `None` it returns `[]` before any check.

Preconditions asserted: `CTS_ESTABLISHED(0,1)` has `(idx, confirmed_at) == (9, 10)`, and there is exactly one POI
`(0,1)`, at IC 7.

| # | Test | Asserts | At HEAD |
|---|---|---|---|
| a | `test_poi_first_activation_is_the_cycle_moment_not_the_anchor` | history `[(10,T,initial),(12,F,imbalance_filled)]`; `history[0]["versions"] == ["V30"]`; `confirmed_idx` 10; end `(14,"reversal")`; status "ended"; **`meta["cts_established_idx"] == 10`** (the moment) | FAILS (9, 9) |
| b | `test_poi_never_activates_before_its_cycle_lifecycle_start` | invariant over every POI: `history[0].idx >= compute_cycle_lifecycle(events, compute_reversal_idx_by_sid(events), floor, cap)[(sid,cyc)][0]`, using the run's own floor and cap; also over `_make_multicycle_data()` | FAILS (9 < 10; multicycle passes) |
| c | `test_poi_first_activation_tracks_confirmed_at_exactly` | set `confirmed_at = 11` on CTS_EST(0,1) **and** BOS(0,1), keeping the definitional identity → `history[0] == (11,T)` and `meta["cts_established_idx"] == 11`. Separates "reads confirmed_at" from "idx+1" | FAILS (9) |
| d | `test_poi_unchanged_when_anchor_equals_moment` | `_make_multicycle_data()` POI (0,2) IC 12: first activation `(15,T)` (only the first — a full-history pin would couple it to the queued imbalance-at-c3 change), end `(20,"next_cycle")`, `cts_established_idx == confirmed_at` | passes |
| e | `test_poi_floor_vs_moment` (table-driven) | floor 9 (≤ anchor: the moment decides, the IC-678 analogue) → `[(10,T),(12,F)]`; floor 10 (== moment: masked, the sub-3 analogue) → the same; floor 11 (> moment: the floor decides) → `[(11,T),(12,F)]`; **`cts_established_idx == 10` in every case** (the masked case's only delta is the meta, as on the replay) | all 3 FAIL (floor 9 on the history; floors 10/11 on the meta) |
| f | `test_poi_of_cycle_collapsed_at_its_moment_never_activates` | `lifecycle_cap=10` → `[]`, `confirmed_idx` None, "inactive"; `cap=11` → `[(10,T)]`. Pass `cap_reason` explicitly and do not assert the dead default `"lifecycle_end"` | FAILS (`[(9,T)]` "ended") |
| g | `test_poi_lookup_reads_confirmed_at_without_fallback` | deep-copy the events and pop `confirmed_at` from CTS_EST(0,1); `monkeypatch.setattr(engine_v2.zones.poi_zones, "compute_cycle_lifecycle", lambda *a, **k: {})`; pass the run's `fib_tracker`; `pytest.raises(KeyError, match="confirmed_at")`. Fails again if someone writes `.get("confirmed_at", ev.idx)` | FAILS (no raise) |
| g2 | `test_derive_poi_zones_raises_on_cts_established_without_confirmed_at` | no stub; `pytest.raises(AssertionError, match="lacks meta['confirmed_at']")` pins the upstream assert; skipped under `python -O` | passes |
| h | `test_activation_applies_pre_window_cts_state` | unit test of `_compute_poi_activation_history` on a 6–8 candle df: one unfilled sd imbalance entering before `first_active`, a CTS_ESTABLISHED at `first_active−1` → active AT `first_active` with non-empty versions. Pins the pre-window loop, which is now load-bearing (a mutant without the loop returns `[]`). Keyword `cts_established_idx=`, unchanged before and after | passes |
| i | `test_fallback_cycle_without_cts_established_keeps_fib_anchor` | a cycle with its CTS_ESTABLISHED removed: `meta["cts_established_idx"] == fib_state.cts_idx`; history `[]` / "inactive" (pins decision 3 until the §7.3 follow-up changes it on purpose) | passes |

**In `tests/test_render_sub_projection.py`,** add `test_poi_first_activation_at_or_after_sub_start`, the POI twin of the
KL R6 pin:
- **Do NOT use the R6 window.** Under cap 85 the only fib is active and unlocked with an end, so the
  `poi_zones.py:447` gate yields zero POIs and the test would pass vacuously.
- Use `_two_lens_records` with an OPEN window, `_set_lifecycle(sub, _START, None, None)`.
- Assert `res.poi_zones` is non-empty, and in every lens `len(attrs["poi_zones"]) == len(res.poi_zones) >= 1` with at
  least one non-empty `activation_history`.
- Assert `history[0]["idx"] >= sub.start_idx`. Expected: 1 POI (IC 63, cycle 0), `[(75,T)]`.
- Floor-decides case: `_set_lifecycle(sub, 78, None, None)` → `[(78,T)]` in both lenses (the raw activation is 75). This
  pins that the sub's `start_idx` reaches the POI floor through the projection.
- It passes under both rules (the floor does not bind at start 60: IC 63 and the raw activation 75 are after it). It
  pins non-vacuity and the mirror's rebase of `cts_established_idx` (== the geometry's cycle-0 moment + `slice_begin`).
- **Added by the landing cold review:** `test_poi_cycle_term_is_the_moment_through_the_projection` — move the geometry's
  cycle-0 moment 2 candles past its anchor (CTS_ESTABLISHED + BOS_CONFIRMED together); the mirrored
  `cts_established_idx` follows the moment, the first activation stays `(63, 75)`. The replay's sub-3 case (a meta-only
  re-value) through projection + mirror; FAILS under the old rule.

**Also:** add `activation_history` (idx, active, reason, versions), `confirmed_idx`, `cts_established_idx` and
`current_versions` to `_poi_sig` in `test_pooled_structure_build.py`, so the dedup-equivalence tests cover them.
Test-only, non-behavioural.

Expected suite: 722 + every new test passes + 1 strict xfail. The exact count is recorded at landing.

## 6. Docs that move in the SAME commit (implement-against-docs)

**POI_ZONES_SPEC §4 "Activation floor"**
- The formula keeps its letters, `first_active = max(cts_established_idx, ic_idx, lifecycle_floor_idx)`. It now defines
  `cts_established_idx = CTS_ESTABLISHED.meta["confirmed_at"]` (the moment).
- Add: "`max(cts_established_idx, lifecycle_floor_idx)` equals the cycle's clamped lifecycle-start
  (`compute_cycle_lifecycle`) whenever the cycle has a CTS_ESTABLISHED. With none, the cycle term falls back to
  `fib_state.cts_idx`."
- The "Open inconsistency" box is REPLACED by a dated "Resolved (Plan D, 2026-09-23)" note.
- The field list gains `cts_established_idx`: "the cycle's CTS-established MOMENT (`CTS_ESTABLISHED.meta
  ["confirmed_at"]`), the activation floor's cycle term.
  - **Meaning changed by Plan D:** saves before it hold `CTS_ESTABLISHED.idx` (the CTS anchor) under this key.
  - Fallback when the cycle has no CTS_ESTABLISHED: `fib_state.cts_idx`, the fib's CTS anchor, not a moment — a known
    naming-standard exception. Live case: counter sub 5 cycle 1, IC 3654 → 3806.
  - Rebased to entity-absolute on sub POIs by the mirror (`_ZONE_META_IDX_KEYS`)."
- `bos_idx` / `cts_idx` are NOT added. They are slice-local on sub POIs and belong to the parked export-hygiene item.

**ARCHITECTURE "Activation floor" NB**, rewritten in place:
- Keep the KL sentence: KL's term is the zone's `confirmed_idx` (= `BOS_CONFIRMED.meta["confirmed_at"]`, the moment),
  clamped to the per-sid structure start.
- Replace the POI sentence: POI's term is `ic_idx`, the cycle term `cts_established_idx` =
  `CTS_ESTABLISHED.meta["confirmed_at"]` (Plan D), and the `struct_start_by_sid` floor. Exception: a (sid, cycle) with no
  CTS_ESTABLISHED has no cycle-lifecycle entry and falls back to `fib_state.cts_idx` (POI_ZONES_SPEC §4).
- In the `ev.idx` table's "Known sites that still read `ev.idx` as a time" list, REPLACE the POI-gate item with: "the POI
  activation sweep's `CTS_UPDATED` transitions at `ev.idx` (pre-window and in-window; the pattern path records no
  moment — an event-contract change, parked)". That list stays the ONE known-sites list.

**PART4 §17.6 "As landed (2026-09-20)" blockquote**, reconciled IN PLACE:
- A dated marker goes FIRST (PART4's convention, cf. :1051/:2023/:2036): "**Resolved for POI — Plan D, 2026-09-23:**
  POI's floor is now `max(cts_established_idx, ic_idx, lifecycle_floor_idx)` with `cts_established_idx =
  CTS_ESTABLISHED.meta["confirmed_at"]` (fallback `fib_state.cts_idx`). The POI meta key `cts_established_idx` now
  holds the moment (it held `.idx`). Measured: sub 0 cycle 2 IC 678 activation 1223 → 1224; 3 meta cells re-valued
  [filled in after step 4]. KL's start side is unchanged."
- The stale POI sentences move to the past tense or get "(until Plan D)". "decide in a zones pass…" becomes "resolved
  by Plan D".

**Other PART4 and PLAN_C spots:**
- **PART4 ~:1099-1107** (the "Phase 3" paragraph): a dated as-landed marker, since it was true then. **~:1112**
  "else ≤1 candle earlier": the marker says this is the observed lag, not a bound.
- **PART4 ~:1239-1243**: "the ≤1-candle extreme-vs-breakout divergence … accepted" has been stale since Plan C (a BOS
  zone's confirm IS the moment). Correct it in place. ~:1235-1238 ("Nothing inherits cycle start") is correct and stays.
- **PLAN_C ~:228 / ~:766** "the KL/POI `confirmed_idx` clamp follows": add an as-landed note that only the ENDS
  followed, and that the POI start moved in Plan D. In the same ranges:
  - ~:229 / ~:769 "three sub cycles" → 3 lens rows = 2 unique cycles;
  - ~:232-233 "lists BOS_CONFIRMED as the only extreme-not-apply exception … fix in §11" → superseded by the
    ARCHITECTURE table;
  - ~:242 "0 of 5 on H1 only by luck" → anchor == moment is the common case (31/34).

**GOTCHAS**, one line under the `ev.idx` entry's "Bugs caused by this": "POI activation gate read `CTS_ESTABLISHED.idx`
(the CTS anchor) under the moment name `cts_established_idx`, so a POI went live up to 5 candles before its cycle existed
(M15 conf sub 0 cyc 2, IC 678: 1223→1224). Fixed by Plan D, 2026-09-23."

**No new LANDMINES entry.** The rule is already in the ARCHITECTURE table and the GLOSSARY Naming Standard (a timing
read uses the moment).

**Memory, after step 4 only** (measured values, never before the replay):
- `project_item_3_poi_lifecycle.md` ~:36 (the scan-start formula);
- `reference_pool_redesign_groundtruth.md` ~:93-95: append "(at Plan C; moved to the moment by Plan D `<hash>`)";
- the pool handoff: BOTH mentions (the STATUS "candidates" line and the residual);
- MEMORY.md: ACTIVE THREAD and the Plan C residuals list;
- `project_zones_timing_audit_20260922.md`: as landed.

**Wording discipline.** Docs written in step 3 phrase every measured value as "expected: …", filled in only after
step 4 (`feedback_spec_writing_precision` rule 8). "2026-09-23" becomes the landing date.

## 7. Out of scope

**Queued (each its own `/compare`, in this order after Plan D):**
1. **Imbalance knowable at c3, not c2.** `poi_zones.py` ~:896 (HEAD `:920`) enters an imbalance at `inst.start_idx`, the middle candle.
   NEXT. Measured: 4 POI rows (H1 sid 1 cyc 2 IC 865/860: re-activations 953→954 and 997→998; M15 sub 7 IC 4048 in both
   lenses: 4118→4119). It does not interact with this fix on the replay (measured); unit test (d) pins only the first
   activation, so it is unaffected too.
2. **The naming / event-convention project**, written up as its own plan (Plan E) with per-stage predictions and a cold
   review. Stages:
   - the pattern-anchor rename + the event-contract amendment;
   - explicit anchor fields;
   - the timing fixes, including **FibTracker timing on the anchor** (activated_at, the fill-check time, the previous
     cycle's `new_cycle` terminal; measured 4 fib_lifecycle rows, sub 3 in both lenses);
   - the CTS_ESTABLISHED + BOS_CONFIRMED flip;
   - the remaining renames.

   Inputs (every site list, design constraint, open question): `engine_v2/plans/PLAN_E_inputs.md`.
3. **The never-established-cycle fallback POI** (counter sub 5, cycle 1) is drawn past its sub's end and carries a fib
   anchor under a moment name. Skipping such POIs would give: counter CSV 13→12, shapes 125→124, traces 153→151.

**Parked (0 delta on this window; not queued):**
- the pattern-path `CTS_UPDATED` moment field (event contract, Plan E);
- the pool's `knowable_at_idx` and sibling clip (§17.12);
- `poi_active_as_of` treats the end as inclusive while the scan is exclusive;
- the in-flight MS proximity gate `i > st.cts.idx`;
- the orchestrator's `(idx, type)` sort;
- the prev-BOS line end;
- `_events_to_structure_levels`;
- mixed slice-local / absolute idx in exported POI `bos_idx` / `cts_idx` and sub fib meta;
- POI_ZONES_SPEC §3.2 IC idx constraints not implemented;
- the dead `scenario_context`.

## 8. Procedure

1. Write tests (a)–(i) and the render pin first. Run them and confirm the "At HEAD" column exactly.
2. Code + comments (§3). The full suite passes (722 + new + 1 xfail).
3. Docs (§6) in the same working tree, with measured values written as "expected: …".
4. Replay (`python -m engine_v2.run_replay > run.log 2>&1` from the repo root) → `/compare` against the baseline.
   Checks, in order:
   1. the fetch gate;
   2. the silent-skip grep, with every line explained against the §4 run.log row;
   3. the 24 CSVs against the §4 table, cell for cell;
   4. chart counts;
   5. a figure-level diff of baseline vs current HTML (`engine_v2.debug.chart_census.load_fig`): exactly the 2 shapes and
      2 hover-trace customdata on confluence, and identical H1 and counter figures.

   Then display the replay timing. Then the documentation checkpoint: fill in the "expected" values, write the memory
   updates, reconcile `_INBOX.md`.
5. Cold review of the landing (adversarial, before the commit).
6. **PAUSE for the user's chart review**: the confluence chart, 2025-12-03 ~15:45–16:00, sub 0's IC-678 POI; hovering
   shows `confirmed_idx=1224`. Then a checkpoint recording the user's verdict.
7. Commit + `/commit-save` on the user's go-ahead. Nothing is re-baselined; the counts are unchanged.

## 9. As landed (2026-09-23, `0a4eadc`) — every §4 prediction held

- **Tests:** written first. At HEAD the §5 "At HEAD" column was confirmed exactly: 6 failed, each for the predicted
  reason (activation at the anchor 9 where the moment 10 is expected; `DID NOT RAISE KeyError` for (g)). After the
  landing cold review strengthened (d)/(e)/g2 and added the moved-moment render pin: **739 passed + 1 strict xfail**
  (722 + 17 new: 14 in `test_poi_activation_moment.py`, 3 render pins). Mutation check: reverting the lookup to
  `int(cts_event.idx)` fails 9 of them (all of (e) now included, via the meta).
- **Code:** `poi_zones.py` lookup reads `int(cts_event.meta["confirmed_at"])` (fallback unchanged); comments per §3;
  `structure_lifecycle.py` docstring only.
- **`/compare` vs `20260922_195430_aadb887`** (reuse of the replay run on this tree; replay 45.9 s wall):
  - fetch gate PASS;
  - silent-skip grep: 21 lines, all explained (5 pandas FutureWarnings; `pending=0`; 5 parent-table lines — (0,0),
    (0,1), (1,0), (1,1), (1,2) — + the two degenerate (1,0)/(1,1) warnings; the 4 `[sweep] UNRESOLVED` degenerate rows +
    their 4 `[pool]` echoes);
  - 3 `[probe_cache]` APPROX hits unchanged; no `REF-ZONE DIFFERS`;
  - **22/24 CSVs byte-identical**. The two POI CSVs differ in exactly the §4 cells: confluence line 4 (`confirmed_idx`,
    `activation_history` idx, meta `cts_established_idx`, all 1223 → 1224), confluence line 19 and counter line 4
    (meta `cts_established_idx` 2828 → 2829);
  - `_swings.csv` present without a baseline: a stale 2026-02-09 leftover the replay does not write — not a finding;
  - **figure-level:** H1 and counter figures identical; confluence = shapes 25 (fill x0) and 27 (confirm line x0/x1)
    15:45 → 16:00 + traces 43/44 `customdata[4]` 1223 → 1224 on all 1044 points; counts 85/245, 153/125, 294/233.
- **Landing cold review** (3 lenses, each non-nit finding adversarially verified; 12 agents): no correctness defect;
  8 minor findings survived (the false `ic_idx <= anchor` comment; tests (d)/(e) and the render pins strengthened; a
  moved-moment render pin added; ARCHITECTURE known-sites list, a PART4 present-tense residual, the §9 line count) —
  all fixed in the same commit, with the nits (comment dating/qualifiers, `current_versions` added to the POI field
  list, vocabulary). Code diff after the review: still the one behavioural line; the replay and `/compare` above stand.
- **Chart review (2026-09-23): the user approved ("looks good").** Committed + `/commit-save`d the same day.
