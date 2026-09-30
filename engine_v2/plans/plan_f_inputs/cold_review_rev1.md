# Plan F cold review, consolidated (HEAD `754a642`)

## 1. Must fix before code

**M1. The design splits MS and FibTracker on cond1 and fib activation, and the plan says they agree** (§2 last bullet, §5, §6 LANDMINES `:188-223`, §7). Reviewer 3 rated this must_fix and reviewer 1 should_fix; I promoted it to must_fix.
- **How MS decides.** The MS in-flight resolver calls `select_fib_anchor_for_cycle` with no moment (`poi_zones.py:1052-1063`, which plan §3.5 keeps). It builds `FibState(active=True)` (`:1084-1094`) and its only gate is `find_ic_candidates` cond3, which is not cut (`:229-236`, `:1098`).
- **How FibTracker decides.** FibTracker cuts at `:536` (`confirmed_at`) and `:880`.
- **When MS uses it.** The MS reader is gated only by `i > st.cts.idx` (`market_structure.py:1975-1981`).
- **H1 has no second chance.** Activation for cycle ≥ 1 happens once. `on_cts_updated` returns None when the key is missing (`fib_tracker.py:1246-1251`, `:1284-1288`).
- **Today.** Non-empty MS inners imply an unfilled sd gap in `(IC, CTS]`, which implies FibTracker `has_unfilled`. So the two layers agree.
- **After Plan F.** Take a lag-0 EST or a raw UPDATED whose only sd gap has c2 == the event candle. MS keeps the POI inner, but FibTracker creates no fib, so no POI is rendered. An `sd_zone_proximity` CTS_CONFIRMED can then fire against a POI that is not drawn. That is the regression signature at LANDMINES.md:215-222 and in memory `feedback_in_flight_vs_downstream_resolver`.
- **Count on this window: 0.** The ms variant is 24/24, the fib variant leaves H1 unchanged, and the only R1 flip at `cross_cycle_fib.py:117` is THRESHOLD@235.
- **Edits:**
  - **§2:** change the claim to "…keeps **cond2** in agreement with the MS mirror." Then add: "cond1/cond3 and fib activation do NOT agree. In the drop case (a lag-0 CTS_ESTABLISHED or a raw CTS_UPDATED whose only sd gap has c2 == the event candle), MS keeps a POI inner that FibTracker never creates. This is permanent on H1 cycle ≥ 1 and lasts until the next CTS_UPDATED on subs. It follows from the two decided items, MS unchanged and drop-not-delay. 0 cases on this window."
  - **§6:** LANDMINES `:188-223` records this as a documented exception to "both layers agree". Add the same note to the `compute_poi_inners_for_cycle` and `select_fib_anchor_for_cycle` docstrings and to the `fib_tracker.py:870-875` comment.
  - **§5:** add a pin: a lag-0 cycle-1 EST whose only sd gap has c2 == CTS_1. `compute_poi_inners_for_cycle` returns a non-empty list; FibTracker has no cycle-1 fib.
  - **§7 re-ask bullet:** "…also closes this MS/FibTracker divergence."
  - Flag it for the user to acknowledge at go.

**M2. The CTS_UPDATED `via` read and the fixtures are under-specified** (§3.5, §5 Fixtures, §8 step 2). Raised by all four reviewers.
- I re-ran the suite with a pytest plugin that wraps the handlers with direct-index reads (no repo edits):
  - Direct `meta["confirmed_at"]` on EST: **20 failures**, 11 in `test_cross_cycle_fib.py` and 9 in `test_main_versioned_cross.py`.
  - Direct `meta["via"]` on UPDATED: **8 failures**, a subset of the 20:
    - `test_cross_cycle_fib.py`: `:102`, `:118`, `:267`, `:306`, `:340`
    - `test_main_versioned_cross.py`: `:104`, `:119`, `:155`
- Neither `_ev` helper sets these keys (`test_cross_cycle_fib.py:40-46`, `test_main_versioned_cross.py:38-42`). Production always sets `via`: pattern name at `market_structure.py:1551`; `"replay_raw"` at `:1716`/`:1722` from the sole caller `:638`.
- Fixture gaps sit at 13/15/23/35/55 and the event idx at 20/25/40/45/60/65. So lag 0 plus `replay_raw` flips no pinned assertion.
- **§3.5 edit:** "`event.meta["via"]` by direct index (the same contract stance as `confirmed_at`)."
- **§5 Fixtures edit:** "`_ev` helpers: CTS_ESTABLISHED gets `confirmed_at = idx` (lag 0). CTS_UPDATED gets `via = "replay_raw"`. A pattern-path test passes a pattern name (e.g. `"continuous"`) explicitly. This affects 20 EST-driven tests, 8 of which also drive CTS_UPDATED."
- **§8 step 2 edit:** "fixtures gain `confirmed_at` (EST) and `via` (CTS_UPDATED)".

## 2. Should fix

**S1. §4 provenance.** No sweep+fib replay exists. PLAN_F_inputs.md:67-75 lists sweep, ms, fib, poiid, sweep+ms+fib, sweep+ms+fib+poiid.
- Edit: "(§2.2: the sweep-only and fib-only runs. sweep+ms+fib is their exact union and ms-only is 24/24, so sweep+fib is derived, not replayed.)"

**S2. §4 Charts, §4 Log and §8 step 6 are wrong in places.**
- **M15 fibs are never drawn:** `export_m15_chart.py:109` sets `"fib": {"lines": False}`, and `:1098` leaves `sid_fibs` unused (`noqa F841`).
- **Where 151/124 comes from:** in the baseline counter HTML, the IC 3654 cycle-1 POI is traces #49 and #50 (status `inactive`) plus shape #29 (a rect to chart end). That is −2 traces and −1 shape, so the count prediction is right but its explanation is not.
- **Moves the plan does not list:**
  - the H1-overlay stretches and confirm lines at H1 997 on the M15 charts (counter shapes #108/#112/#115/#119 at 2026-01-13T15:45);
  - the IC 4048 hover 4118 (traces #74/#75).
- **Log lines the plan does not list:** in the counter sub-5 projection, `[fib_tracker] total fibs=` (`orchestrator.py:262`) and `[poi_zones] total=` (`:318`) each drop by 1. These would trip "anything else = STOP".
- **Edits:**
  - Charts: "(−2 M15 POI hover traces and −1 outline rect: the removed IC 3654 cycle-1 POI. M15 fibs are not rendered.)"
  - Delete "sub 5 cycle-0 fib lines extend to 4083".
  - Add to "moved": the H1-overlay 953→954 and 997→998 on both M15 charts, and hover 4118→4119.
  - Log: add the two −1 totals lines and the `CROSS (0->1) v0 ACTIVATED` line that disappears.
  - §8 step 6: "…without the phantom cycle-1 POI (the dropped fib is visible only in the fib_lifecycle CSV)".

**S3. §5 header and §8 step 2 ("each fails on HEAD") cannot be met.** These pass on HEAD, because HEAD cuts nothing (`imbalance.py:159-166`):
- the lag-1 EST that still activates;
- pattern-path with no cut;
- the c0 cache being True;
- the default/None cases that stay unchanged;
- merged `(1,3)` at `evaluated_at=2`.

Edit: split §5 into **behaviour tests** (fail on HEAD) and **guard pins** (pass on HEAD). For each guard, name the wrong variant it catches: a cut keyed on `check_to`, R2, a cut on pattern-path, a filtered c0 cache. Reword §8 step 2 to match.

**S4. The §3.5 "handler-scoped state" rationale is false.** Tests call only `_get_latest_cross` and `_cross_allowed_for_target`, and neither reads imbalances (`fib_tracker.py:1921-1950`, `:2011-2021`). Also, `None` means both "outside a handler" and "no recorded moment".
- Edit: delete "tests call several private methods directly".
- Implement the scope as one context manager that restores the previous value rather than resetting to `None`, and asserts it is not nested. No nested handler calls exist today: there is no `self.on_cts_` in fib_tracker.py.
- Make `_event_moment` raise on an unexpected event type.

**S5. The raw-path set `{"replay_raw","raw"}` duplicates the emitter's literals.**
- `"raw"` is only the unused default at `market_structure.py:1701`; the sole caller `:638` passes `"replay_raw"`.
- The canonical definition is `via == "replay_raw"` (ARCHITECTURE.md:100, GLOSSARY.md:82).
- A future `via` value (e.g. the commented-out `range_expand` at `:1253`) would silently get no cut.
- Other code already encodes the same "moment" column: `knowable_at_idx` (`sub_structure_pool.py:84-102`, where EST → `ev.idx` is pinned at `test_sub_structure_pool.py:564`) and `_second_cts_moment` (`unified_probe.py:450-456`, a `.get` fallback).
- Edits:
  - Define `CTS_UPDATED_RAW_VIA = "replay_raw"` in market_structure.py. Use it at `:638` and remove the dead `"raw"` default.
  - `_event_moment` compares against the constant, ideally as a public `event_moment` next to `StructureEvent`.
  - §7: "knowable_at_idx and _second_cts_moment converge onto it in Plan E E3b."
  - Add a unit test: an MS-emitted raw UPDATED is classified as a moment and a pattern UPDATED is not.

**S6. The `evaluated_at=None` default silently means "no cut"**, which is the kind of defect this plan fixes. `select_fib_anchor_for_cycle` is called positionally by both layers (`fib_tracker.py:880-891`, `poi_zones.py:1052-1063`), and the in-flight caller swallows every exception (`poi_zones.py:1116-1117`; GOTCHAS "Positional resolver args drift").
- Edit: make it keyword-only and required on all three functions.
- Sites that intentionally skip the cut pass `evaluated_at=None` explicitly, each with a one-line reason: `poi_zones.py:229`, `market_structure.py:2050`, the `compute_poi_inners_for_cycle` select call, and the two FibTracker cycle-0 cache writes.
- Output is byte-identical. The cost is updating `test_imbalance.py:336/:340/:341/:358` and `test_cross_cycle_fib_routine.py:61/:74/:204/:221`.
- This is hardening: adopt it or state why not.

**S7. The Plan E collision is not annotated.**
- PLAN_E_inputs.md:656-661, item 2: "The split: `current_candle` (moment) / `anchor_idx`".
- PLAN_E_inputs.md:504-505: `current_candle` "values become moments (E3a)".

After E3a, `current_candle` and `evaluated_at` would carry the same value. Edit: add both to the §6 annotation list: "E3a keeps one moment parameter, `evaluated_at`; `current_candle` is renamed to its fill-horizon role or folded in." Put the same note in the CROSS_CYCLE_FIB_SPEC `:90-95`/`:133-135` edit.

**S8. The fallback-POI docs go stale and are not in §6.**
- POI_ZONES_SPEC.md:393-397 names the live case "counter sub 5 cycle 1, IC 3654: value 3806", which Plan F removes.
- PLAN_E_inputs.md:816 predicts the post-E item as counter 13→12, 125→124, 153→151, exactly Plan F's delta. See also `:888` row 19.
- PLAN_D_poi_activation_moment.md:271-272 carries the same numbers.

Edit: add these lines to §6: "no live case on this window since Plan F; the general defect stays parked; the post-E predicted delta is now 0 here."

**S9. Skill and memory values that move are not in §6.**
- `.claude/skills/compare/SKILL.md:470` ("`M15.counter` chart: traces=153, shapes=125") and its history paragraph `:473-490`.
- MEMORY.md:12 and `project_sub_structure_pool_architecture.md:81` (counter 153/125, tests 739).

Edit: move them to 151/124 and 739+N in the same commit, per the root CLAUDE.md rule.

## 3. Nits

- **§3.5 row `:1318`.** `c0` can be `{}` at `fib_tracker.py:1329` and `:1350`, because `:1347` sets Scenario 1 without checking `c0`. State the form: "absent c0 → False (today's default); else direct index." Unreachable on this window.
- **§3.2 naming.** Rename `formed_end` → `formed_end_idx`. Consider an `ImbalanceInstance.overlaps_formed_prefix(...)` method next to `overlaps` (`types.py:139-141`) instead of an inline copy. Name `c0_has_unfilled` so it says "uncut".
- **§4 parenthetical mixes index spaces.** Use: "EST local 53 = abs 2368; raw UPDATED local 61 = abs 2650; THRESHOLD local 235 = abs 3806." Fib meta `activated_at`/`reactivated_at`/`locked_at` stay slice-local; only `deactivated_at`, `start_idx`/`end_idx` and `bos_idx`/`cts_idx` are shifted (`entity_df_mutation.py:274-289`).
- **§7 pattern-path bullet.** A pattern-path duplicate at the same idx (ARCHITECTURE.md:100, 1 of 37 rows) is processed right after the raw row, because the sort is stable (`orchestrator.py:152`). It has no moment, so it re-asks without the cut at `:1183`/`:1330`/`:1351`. That is correct in knowledge terms, since its apply candle is at least idx+1, and 0 cells change on this window. Add one sentence; the §5 raw `:1183` test should have no such duplicate.
- **§5 `_update_fib_cts` test is under-specified.** Spell it out: a raw UPDATED at t whose only unfilled sd gap has c2 == t now deactivates at t with reason `all_imbalances_filled` (`fib_tracker.py:1470-1473`), which now means "no formed unfilled imbalance". It reactivates at the next raw UPDATED. Note this in FIB_LIFECYCLE_SPEC.
- **Explainability.** The "no unfilled imbalance" NOT-activated prints cannot tell "not yet formed" apart from "no gap". Add `evaluated_at` to them.
- **ARCHITECTURE.** `:106-109` says "Not audited: … `*_THRESHOLD_UPDATED`", which contradicts the new row; narrow it to BOS_THRESHOLD_UPDATED. Add FibTracker to the direct-index `confirmed_at` readers (`:129-134`). Update the "not yet measured" text at `:135-142`.
- **Extra doc lines:**
  - POI_ZONES_SPEC `:137-143`, `:222-226`;
  - GOTCHAS `:1563-1566` (the late-activation fix is now cut);
  - PRE_REFACTOR_INVARIANTS `:221-233`;
  - the `select_fib_anchor_for_cycle` docstring ("cond1 … as of cts_idx");
  - a cross-link at CROSS_CYCLE_FIB_SPEC `:66`.
- **Relabel.** The `test_poi_activation_moment.py:157-159` comment ("Pinning the full history would couple…") is superseded by the new IC-12 pin. GOTCHAS.md:1405 quotes the pre-c3 history.
- **§6 memory.** Drain `memory/_INBOX.md`, as PLAN_F_inputs.md:223 asks, and file the Windows >260-char path gotcha (`_INBOX.md:43-44`).

## 4. Refuted reviewer claims

- **Reviewer 4's S9 wording** ("−1 POI trace/shape and −1 fib trace") is wrong. The baseline counter HTML shows −2 POI hover traces (#49, #50) and −1 rect (#29), and M15 charts have no fib traces (`export_m15_chart.py:1098`).
- **Reviewer 4's relabel nit:** LANDMINES.md:1569 does not quote the history `[905:A, 952:D, …]`. It cites `confirmed_idx` 997 inside a dated 2026-05-26 incident narrative, so annotating it is optional.
- **Severity:** reviewer 1 filed the divergence as should_fix; it is promoted to M1. Reviewer 2's must_fix and the three should_fix copies of the `via` item are merged into M2.

## 5. Verified OK

- The §3.2 filter equals the `variant_run.py` cut, and `evaluated_at=None` is byte-identical to `imbalance.py:159-166`.
- The sweep identity with `evaluated_at=t` is exact: the IC is a −sd candle, and c2 matches the gap direction.
- fib_tracker.py has exactly 11 production `has_unfilled_imbalance` calls (536, 1183, 1318, 1436, 1445, 1451, 1462, 1527, 1535, 1540, 1571), as §3.5 says.
- Only the three named handlers reach an imbalance read. Dispatch is at `orchestrator.py:204-239`, and EST is gated on `bos_by_cycle`.
- Every production CTS_ESTABLISHED has `confirmed_at` and every CTS_UPDATED has `via`. CTS_THRESHOLD_UPDATED idx is the processing candle.
- Index spaces are consistent on the H1 and sub paths: slice-local events and slice-local instances.
- The harness differences D1–D4 change 0 cells. The flipped sub-2 and sub-3 updates are `replay_raw`.
- `dead_cycles` cannot latch wrongly: every walked window ends at CTS_k < conf_k ≤ the moment.
- The cycle-0 cache must stay uncut for cond2, so the split at `:536`/`:1318` is correct.
- The §4 CSV cells match the baseline save, and the counter count of 151/124 is right.
- The IC-12 prediction reproduces: `[(15,A)]` → `[(15,A),(19,D,imbalance_filled)]`.
- `formed_at` as a property on the frozen dataclass changes no CSV or contract, and the names follow the GLOSSARY Naming Standard.
- The §6 line references are accurate (reviewer 4's check).