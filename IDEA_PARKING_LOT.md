# Deferred-Items Register & Idea Parking Lot
Multi-Timeframe Forex Trading System

**The ONE canonical list of everything deferred, parked, "revisit later", or planned-but-unbuilt.** Each entry is one
line: what, its trigger or target week, and a pointer to where the DETAIL lives (a spec section, a plan, a memory file —
the detail is never duplicated here). Ideas are recorded immediately but implemented deliberately; do not implement an
entry outside the syllabus sequence without the user's go.

**How to add / close an entry.** A new deferral: one line here (bucket, trigger, source) + its detail in the owning doc.
Closing one: move its line to "Recently closed" with the date and commit — never delete silently. Memory
(`MEMORY.md` "Next session priorities") names only what is NEXT and points here for the rest.

**Built 2026-09-30** from a full sweep (every repo `.md`, all 51 memory files, code TODO/FIXME comments; grep for
deferral wording, each hit read in context, uncertain ones checked against the code). An item phrased without deferral
wording could still be missing — add it when found. Memory paths are relative to the Claude project memory folder
(`~/.claude/projects/<project>/memory/`).

Status: **OPEN** (to do) · **DECIDE** (needs the user's call first) · **TRIGGER** (waits for a condition) ·
**CONFIRM** (possibly superseded — check with the user).

---

## A. Week 8 close-out — Week 8 DoD SIGNED OFF by the user 2026-09-30 (after bucket C landed); merged to `main`

**Week 8 is done.** The user held Week 8 open until bucket C (Part 4 closure) landed. The DoD itself ("HTF and LTF
context aligns logically; confluence vs non-confluence is visually obvious") was already met. C landed 2026-09-30, and
the same day the user signed off the DoD; `/commit-save` and the merge `week8-volmom-multitf` → `main` followed.
The open lines below carry over (fold in when their files are next touched); D and later come after the merge.

**Chart review with the user** — all items closed 2026-09-30 (see "Recently closed").

**Hygiene** — the 2026-09-30 pass is done (see "Recently closed"); what is left:
- **Charting audit 2026-09-21 — the style / code items** — #7 sub-native zone outline hardcoded (no `zone.m15.poi.*`
  key), #8 hover-label colours hardcoded, #9 the M15 zone-hover hitbox lines hardcoded (H1 reads the registry), #11 the
  unused `structure.reversal_watch_line` key + the never-plotted `watch_y` series; plus the audit's still-unverified
  list. Also: 8 unused locals in `charting/export_m15_chart.py` (ruff F841, pre-existing). OPEN (fold in when those
  files are next touched). Detail: memory `project_charting_audit_20260921.md`.
- **Two dead reads found by the Week 9 /prepare + item-1 talk** — `_run_multi_tf_dual(poi_zones=...)` is never read
  (`pipeline/orchestrator.py` ~:837; fold into item 1's landing B, which touches the orchestrator's `POIConfig`);
  `wvmi._find_temporary_lp` computes `fp_vol_dir` and never reads it (fold into "The agreed direction for wave candles
  + WVMI", D.2). OPEN.
- **The H1-main frame is classified and stamped "H1" whatever the config timeframe** — `orchestrator.run_pipeline`
  calls `apply_candle_classification(df)` with its default (`timeframe="H1"`), so a re-enabled M15-main window
  (`config.py` keeps them commented out) would get H1's pinbar floor AND the institutional pattern's H1 cap / params
  through the frame's `attrs["timeframe"]` (PLAN_H landing review, 2026-10-06). Pass `CONFIG.timeframe` (or assert H1)
  when such a window comes back. OPEN; not reachable on the current config.
- **Naming-Standard audit of never-audited index keys** — KL `bounds_steps[*].start_idx` (the base ANCHOR on the INIT
  step, a MOMENT on expansion steps — one key, two kinds), RANGE_STARTED `confirm_idx` / `start_idx`, STATE_CHANGED
  `effective_idx`. Parked by the user ("don't start unasked"); cheap; the `bounds_steps` one should land before
  anything reads zone expansions for entries / stops. DECIDE when — raise it when D.1 item 3 (zone size adjustments)
  starts. Detail: `engine_v2/GLOSSARY.md` "Naming Standard".

## B. Week 8 scope — decided 2026-09-30 (the M5 layer moved to D; the syllabus extras dropped — "Recently closed")

## C. Part 4 closure — DONE 2026-09-30 (all three items in "Recently closed"; Part 4 is complete)

## D. Week 9 Part 1 — Entry prerequisites (before Entries)

**Week 9 = "Entries & Decisions"** (renamed by the user 2026-09-30), in three parts: **Part 1** Entry prerequisite
items (this bucket), **Part 2** the actual Entry decision build & logic, **Part 3** Other decisions build & logic
(SYLLABUS Week 9). User: "I agree that these items should be worked on prior to Entries"; a rough outline, details
come from the user as Week 9 reaches each item.

**Order — set by the user 2026-10-01: the heavy items first (D.1), the rest after (D.2).** Verbatim: "I would like
to tackle the heavy lifting items first. These will be ones with extensive documentation, build and changes to
structure elements that could differ a lot from our current iteration. Each of these may require separate design /
planning and build sessions." Items 1 and 3: the user provides updated detailed specs. Items 2 and 4: "may involve
flipping across multiple different scenarios & parameters and reviewing the output/charts to get to the final output
we want." "The remaining items you listed we'll keep in the backlog and decide on them after the above items are
complete." Item 1 opens the next working session.

**D.1 — Ordered (the heavy items):**
1. **One POI per cycle** (redefine the IC & POI zone logic) — NEXT. User 2026-09-30: "will reduce POI zones in each
   cycle to just 1 (vs. currently where a cycle may have multiple POI zones)". **SPEC DECIDED 2026-10-05** (the user's
   spec, six rounds of questions): ONE IC per cycle, chosen by the *institutional pattern* (initial / opp /
   continuation legs, ranks 1A–3D, tiebreakers) among the §3.1 candidates touching the 61.8–80 zone; no pattern → no
   POI. Canonical: `engine_v2/plans/PLAN_H_institutional_pattern_poi.md` (+ `plans/plan_h_inputs/`: the spec copy,
   figures, prototypes). Implementation plan PLAN_H §8 REVIEWED + folded 2026-10-06 (3 lenses, §8.8; the user's R1–R4 in §6);
   landing A (the selector + debug CSVs `*_ip_candidates.csv`, nothing switched) BUILT 2026-10-06, `/compare` == its
   prediction (PLAN_H §8.3a), saved `20261006_120513_e1c597c`; the variant replays V1 / V2 / V3 + the test census
   MEASURED 2026-10-06 and B's prediction written (§8.4a: V3 == the hand cross-check; the 4 never-active POIs are
   collapsed cycles; T1 delays 2 activations; the census 18 FAIL / 99 moved-silent); next: landing B (the switch). The V30 / V60 / V90 code is KEPT unwired (X1); POI_ZONES_SPEC §3.2 stays unapplied
   (D4). Baseline (Week 8 save `20260930_151531_8b11bc2`; one POI row = one IC, carrying 1–3 variants): H1 5 ICs over
   3 cycles; M15 confluence 30 over 19; counter 12 over 7. **What moves when the POIs move** (the user accepted it, D2 /
   C3 — "We won't be able to judge whether those changes are appropriate or correct until I can review the charts";
   PLAN_H §3): H1 sid 1 cycle 2's sd-proximity `CTS_CONFIRMED` at 926 (IC 865 → 832: ≥ 939 or pullback); M15
   confluence sub 1 cycle 1's at 1902 (IC 1889 → 1875: likely survives); the zone-proximity sd triggers H1 sid 0
   cycle 1 710 → 748 and sid 1 cycle 2 926 → 939 — the H1 WVMI gates and counter subs 3 / 5's triggers → 6 of the 11
   trigger records re-derived. Related: G "Cascade-viz hover extras" (obsolete once the variants go).
2. **Single-cycle fib POIs on subs, and revisit the overall sub cross-fib logic** — user 2026-10-01: "Single-cycle fib
   POIs on sub, and also additionally revisit the overall sub cross-fib logic". Today a sub's single-cycle fib is only
   a fallback (cross fails, established phase); the change lets the two coexist — additive, more POIs, so it pulls
   against item 1, which is decided first. Iterative: scenario / parameter flips + chart review. **Also fix here
   (user 2026-10-02, PLAN_H D5): the sub MS snapshot's fib** — a sub's in-flight POI snapshot (sd-proximity CTS
   confirmation) always uses the INTRA-cycle fib (`select_fib_anchor_for_cycle` → `"intra"` for sid 0, every sub)
   while the drawn sub POIs come from the cross fibs: they differ on 8 unique sub cycles (e.g. confluence sub 0 cyc 2:
   drawn POI IC 678 vs MS's fib zone ~100 pips away); no outcome diverged on the window. "I think this IS something we
   do need to fix, but we can do that as item 2." Detail: memory
   `project_sub_single_cycle_fib_pois.md`; `engine_v2/zones/CROSS_CYCLE_FIB_SPEC.md` (§13), `POI_ZONES_SPEC.md`
   "Cross-Cycle Mode".
3. **Zone size adjustments** — user 2026-09-30: "currently some KL & POI zones may be very large or small in their
   current base form. This will cover additional logic that adjust the size." The user provides a detailed spec.
   Raise when it starts: A "Naming-Standard audit" (`bounds_steps[*].start_idx` — land it before anything reads zone
   expansions). Related: "Zone strength scoring" (D.2). Detail: `engine_v2/zones/KL_ZONES_SPEC.md`,
   `POI_ZONES_SPEC.md`; measured sizes: memory `project_week9_part1_inputs.md`.
4. **Pinbar body-pip floor calibration** — scheduled here by the user 2026-10-01 (was TRIGGER: strategy optimization):
   the provisional values (H1 2.2 / M15 1.0 / M5 0.5 pips, NZD_USD only) and the reclassify-to-pinbar approach itself.
   Iterative: parameter flips + chart review; a candle-type change cascades (patterns → structure → zones), so every
   flip gets a full `/compare`. Detail: memory `project_pinbar_body_pip_floor.md`,
   `engine_v2/features/CANDLE_BODY_FLOOR_NOTES.md`.

**D.2 — Backlog: decided after D.1 is complete (user 2026-10-01).**
- **Entry confirmation candle patterns** — user 2026-09-30: "define new candle patterns that will be used to confirm
  entry (when necessary). This is a prerequisite for Entry, but can go last in Part 1 or first in Part 2." Related:
  "Candle-direction / rejection confirmation at a zone touch" below; D.1 item 4 (the pinbar floor); the pattern engine
  (`patterns/structure_patterns.py`, `features/candles_v2.py`). OPEN.
- **The M5 layer** (`H1.main >> M5.counter` — a third lens directly under H1.main, the Part 4 vision's 4th chart;
  memory `project_part4_vision.md`) — DECIDED 2026-09-30: its own Heavy block (plan + cold review), before any Entries
  work that reads M5; not a Week 8 blocker. Footprint seen so far: `debug/export_sub_tables.py` hard-codes `_M15_`, and
  `pipeline/orchestrator.py` (~:646) runs the whole multi-TF path only when `"M15"` is in `lower_timeframes`. Related
  (only if M5 ever nests under an M15 sub): the canonical encoding of a deeper path, e.g. `H1.main >> M15.counter >>
  M5.confluence` (PART4 ~:176 "Open (deferred)"; PART4 §13.6, §17.12). BACKLOG.
- **Zone strength scoring** — syllabus Week 7 (`strength_score` / `strength_flags`, "strong zones look strong", click a
  zone → strength + flags); `KLZone.strength` is always `0.0` (`zones/kl_zones_v1.py`). DECIDE (scope). Idea attached:
  scale strength thresholds by ATR percentile (volatility regimes).
- **Candle-direction / rejection confirmation at a zone touch — as an ENTRY condition** (user 2026-09-30: not a trigger
  filter; triggers stay structural). The May trigger filter (`47e7532`, reverted `2afd3b2`) is the reference form:
  trigger candle body opposing the zone side. DECIDE when Entries are designed. Detail: this register's "Recently
  closed" (the 2026-09-30 measurement).
- **Should subs get the DERIVED CTS KL zone as their probe reference?** (today every sub reference zone is ad-hoc).
  Unscheduled follow-up. DECIDE. Detail: PART4 §17.12, GOTCHAS (the `kl_zones=[]` pool-path note).
- **No-pattern cycles — a fallback POI** (after D.1 item 1; today: no POI, PLAN_H D7): (a) search the lower TF for the
  same institutional pattern (the user's spec: "in future iterations we will build an additional process where we
  would go into a lower TF"; why the leg cap is 3 on H1 / M15 and likely 2 on M5; PLAN_H H2); (b) the kept V30 / V60 /
  V90 selection (PLAN_H X1: "There is still a chance that I use those as fallbacks … we might still delete in the
  future"). DECIDE (after the item-1 chart review — "we will need to come back and understand if losing their POIs in
  these cycles is appropriate").
- **Test the POI_ZONES_SPEC §3.2 scenario IC filters** — never implemented; kept in the spec, unapplied (PLAN_H D4:
  "I still want to keep these and test into them in the future"). Any IC filter stays OUTSIDE the institutional pattern
  and is applied first. DECIDE.
- **The agreed direction for wave candles + WVMI** (PLAN_H Q-W) — item 1 gives the IP legs the *agreed direction*
  (candle and volume direction agree; one at 0 → the other decides; both 0 = a true doji). Wave candles and WVMI keep
  GLOSSARY "qualified candle" (no volume-only-doji row) for now — user 2026-10-05: "we likely will also make the change
  for wave candles & WVMI, but we'll come back to that later". "Qualified" is implemented twice
  (`wave_candles._is_qualified`, inline in `wvmi._find_temporary_lp`) → one helper first; outputs change → its own
  `/compare`. DECIDE (when).

## E. Trigger-based (only when the trigger fires)

- **Cross-cycle fib bundled follow-up** — TRIGGER: a window that produces a MAIN cycle ≥ 2 cross. Then together:
  generalize the MS in-flight POI resolver (`compute_poi_inners_for_cycle`, cycle-1-only) to the full walk ("11b
  Part B"); reconcile the main cross's bespoke meta; widen the config window incrementally to FIND such a window and
  chart-review + `/compare` it. Parked with it (open questions): fill-as-of convergence main → "current"; the
  Scenario-1/revert ↔ ceiling merge; `M = 0` behaviour; the deeper-walk fill-as-of decision. Detail:
  `engine_v2/zones/CROSS_CYCLE_FIB_SPEC.md` §11b + §13, memory `project_cross_cycle_fib_unification.md`.
- **Plan F §7 follow-ups** — re-ask a failed fib decision when a relevant gap's c3 closes ("dropped" → "one candle
  later"); the MS vs FibTracker activation divergence (measure first); latent: `inst.meta` mutated on imbalance
  instances shared across projections, the chart hover picks a POI by nearest inner price, single-candle stretches
  dropped. TRIGGER (chart review shows a dropped fib) / parked by the user. Detail:
  `engine_v2/plans/PLAN_F_imbalance_c3_knowability.md` §7.
- **Double RANGE_STARTED from the offline finalize** — fixture-only, 0 on the reference window. Parked by the user.
  Detail: memory `project_zones_timing_audit_20260922.md`.
- **`REVERSAL_CANDIDATE` straddling a sub's cap** — `knowable_at_idx` keys it on `ev.idx`, not its `apply_idx`, so a
  sub capped by something else (parent end, same-direction replacement) could keep a candidate that applies past the
  cap. 0 on the reference window (2026-09-30). TRIGGER: a window where one is kept with `apply_idx > cap`. Detail:
  PART4 §17.12.
- **Performance** — TRIGGER: runtime hurts. Entity-direct MS compute (no slice + lookback; needs an MS refactor —
  LANDMINES "slice-elimination is a deferred optimization"), same-TF feature sharing by reference (PART4 §9.6),
  parallelism and the remaining pandas reads in MS (memory `project_ms_optimization_opportunity.md`).
- **Mirror initializes new columns with `pd.NA`** instead of `_ensure_output_cols`' per-column defaults — TRIGGER: it
  bites (the slice-copy drop is the safeguard). Detail: LANDMINES ("Slice Copies Inherit Mirrored Structure Cols").
- **Audit positional-index df columns that survive the sub slice** — TRIGGER: adding any df column whose value is a
  positional index. Detail: memory `project_is_range_pollution_fix.md` "Open follow-up".
- **Opposite-direction POIs** — the zone-proximity logic assumes "POI ⟹ sd"; TRIGGER: any feature producing an opp_sd
  POI → revisit `check_zone_proximity`. Detail: `engine_v2/zones/POI_ZONES_SPEC.md` (~:420).

## F. Week 10 — live readiness

- Incremental `is_filled` (per-instance fill state machine) — IMBALANCE_FILL_SEMANTICS "When to revisit";
  FIB_LIFECYCLE_SPEC §13 (live per-candle fib evaluation).
- Incremental imbalance detection (instances created at `formed_at`, grown by explicit extension events) — GOTCHAS
  (the imbalance-prefix note, "When to revisit").
- Per-candle (not snapshot) POI inners in MS — MARKET_STRUCTURE_SPEC (~:350).
- Phase 3 per-candle dual-lens driver (the sweep's step body is its loop body) and live-mode pool GC (evict when the
  parent ended + no live reference) — PART4 §17.12.
- Ad-hoc zone wait-for-candles (a zone whose neighbour candle is past the live edge) — memory
  `project_ad_hoc_zone_wait_for_candles.md`.
- Moment-order event processing + live-mode pattern selection — PLAN_E Q3; memory
  `project_zones_timing_audit_20260922.md` item (b).
- A live / incremental probe (candle-by-candle) — `engine_v2/plans/PLAN_B_double_cts_early_stop.md` §7.
- The WVMI batch `update_temporary_lp` pass is dead in batch mode (creation already searched the same bounded range)
  — it only matters live. Detail: `engine_v2/plans/plan_e_inputs/review_scripts/mutants_plan_g.py` (equivalent N11 /
  N12).

## G. Idea backlog (no slot yet)

- **Fib storage unification** (`_cross_cycle_data` → versioned `_fibs`) — "DEFERRED, not closed (user 2026-05-27):
  may revisit". Detail: `engine_v2/zones/FIB_LIFECYCLE_SPEC.md` §13.
- **Uniform 3-tier chart opacity** (brightest = cycle not ended, mid = sid not ended, faded = sid ended; one
  `opacity_for(sid, cycle)` helper for every element) — its own session. Detail: FIB_LIFECYCLE_SPEC §9.4 + §13.
- **Cascade-viz hover extras** — `current_versions` vs peak; a V90 → V60 → V30 downgrade cue (OBSOLETE once D.1 item 1's
  landing B removes the variants from the POIs — close it there). Detail: memory
  `project_item_5_cascade_viz.md`.
- **Breadcrumb header with cross-chart navigation** — "not in v1". Detail: PART4 §16.9.
- **Split `PART4_REFACTOR_SPEC.md` into canonical spec files** (a new `MULTI_TF_SPEC.md` + sections moved into
  `MARKET_STRUCTURE_SPEC` / `CHARTING_SPEC` / `LANDMINES`) — the drafted §13.7 plan; PARKED as optional by the user
  2026-09-30 (§13.7 was finalized in place instead: ~235 `PART4 §x` citations, no behaviour gained). Detail: PART4
  header "Canonical" + footer.

## H. Deferred indefinitely

- `armed_idx` chart surfacing; consumer-specific fill predicates (IMBALANCE_FILL_SEMANTICS "When to revisit").

## I. Needs confirmation (possibly superseded)

- (none — the last item was closed 2026-09-30; see "Recently closed")

---

## Recently closed (history pointers)

- **`PRE_REFACTOR_INVARIANTS.md` deleted** — 2026-09-30 (PART4 §13 item 7): every item was already canonical or
  superseded (UC1, the three-probe family, the May checkpoints, `meta["active"]`, the 9/6/3 pips) except the
  `cts_phase` values → `MARKET_STRUCTURE_SPEC.md` "State machine overview" (`FALSE_BREAK` declared, never assigned).
  Memory `project_part4_workflow.md` retired; its handoff-prompt practice kept (memory
  `feedback_session_handoff_prompt.md`).
- **§13.7 doc finalization** — done 2026-09-30, IN PLACE (user): PART4 stays the canonical multi-TF spec (status
  header, §13 step statuses, footer); the split moved to G. The §16.10 file-naming deviation resolved as SPEC FOLLOWS
  CODE (`{basename}` + `_{TF}_{lens}` suffix, one flat folder per save — §12 / §16.10 / §16.11 rewritten), with a
  nesting rule for later (every path segment after `H1.main`: `_M15_counter__M5_confluence`).
- **§13.5.e remainder** — CLOSED 2026-09-30 WITHOUT deleting the `s_res.df.attrs[...]` block (user decision): measured,
  the registry's `H1.main` entity IS `s_res.df` and an entity's artifacts live in its df's `attrs` (PART4 §9.2 / §9.3),
  so the block is the registry's store and both charts already read it through the registry. Landed: the stale
  "DEPRECATED" comment relabelled, 2 dead `run_replay.py` writes deleted, PART4 §9.3 / §13.5.e + LANDMINES fixed;
  byte-identical (24 CSVs + figure JSON). PART4 §13.5.e "CLOSED" note.
- **Week 8 hygiene pass** — 2026-09-30, all byte-identical (24 CSVs + figure JSON; only the `a8ed1cc` wave-candle
  hovers differ from the save): the dead imbalance helpers `get_unfilled_imbalances` / `has_imbalance_in_range`
  (`f630640`); the legacy MS functions `compute_structure_scenario_3` / `compute_structure_from_start` /
  `identify_start_scenario_2_after_reversal` + their private helpers + `tests/test_scenario3.py` (`1071347`); the M15
  chart's inert fib toggle + never-applied dot opacity tier — charting-audit #1 / #2 (`cd8eb9e`); the stale-docs sweep
  (proximity 8/5/3 + probe-reset 4/3/2 pip values, Commit 2 wired, (c) done, the Week-6 overview, the §13.5.d
  docstrings, charting-audit #10 + the verified doc-only items) (`45912ac`) + the stale memory notes; `stash@{0}`
  dropped (`6fdb993`). User decisions: register "Week 8 close-out decisions" commit `ee2314e`.
- **Syllabus Week 8 extras** (pair / TF dropdowns, a "show HTF context" checkbox, `ContextSnapshot`) — DROPPED by the
  user 2026-09-30 as superseded by Part 4: one static chart per entity / lens replaces the dropdowns + checkbox, and the
  entity registry holds every snapshot field (HTF trend = the H1.main structure; active HTF zones = the H1 zone attrs;
  LTF candidates = the `TriggerRecord`s / subs). `engine_v2/SYLLABUS.md` Week 8 notes it.
- **"(b) the (0,0,1) start-logic update"** — closed 2026-09-30 on the memory's own evidence: it IS the true-first-
  breakout cycle-0 redesign (design locked 2026-06-07, Commit 1 2026-06-08, Commit 2 2026-06-16) — memory
  `project_unified_identify_start_probe.md` ("the (b) start-logic update is a FULL DESIGN … see
  project-true-first-breakout-cycle0"), `project_true_first_breakout_cycle0.md`.
- **`get_unfilled_imbalances` direction filter** (was H) — dropped 2026-09-30 with the user's "delete" decision (A).
- **Zone-proximity candle-direction filter** — re-discussed 2026-09-30: a no-op on the reference window (all 6 H1
  triggers + both sd-proximity CTS confirmations already point into their zone; in May it cut 48 → 26, before Rules
  1-3); user: no trigger filter — triggers stay structural; the idea moves to D (entries). Filter + revert: `47e7532` /
  `2afd3b2`.
- **POI two-stroke review over the full window** — done 2026-09-30: every POI's on/off history re-derived per candle
  and consistent (`review_scripts/poi_cond3_check.py`); only 2 POIs ever deactivate (H1 s1c2, sub 7 s0c0); the charts
  already draw each active stretch; user: OK, no change. Memory `project_item_3_poi_lifecycle.md`.
- **Half-clipped reversal check** — done 2026-09-30: 0 on the reference window (every kept `REVERSAL_CANDIDATE`
  applies at its own sub's window end); the limit itself stays — bucket E. PART4 §17.12.
- **Wave-candle hover "BOS zone:" on CTS lines** — fixed 2026-09-30 `a8ed1cc`: kind + role label from one builder
  (`wave_candles.wave_candle_hover_lines`); 55 of 110 labels had been wrong. `engine_v2/zones/WAVE_CANDLES_SPEC.md`
  "Rendering simplifications".
- **WVMI pass** — Plan G landed 2026-09-30 (`83c7a12`, fold-in `576bf6f`, Int64 `3781cab`); `engine_v2/plans/PLAN_G_wvmi_unique_sub.md` §9.
- **§13.5.d paired carve-out removal** — subsumed by the sub-structure lifecycle redesign (2026-05-25); leftovers in A.
- **KL lifecycle convention** (`meta["active"]` / `deactivated_by` indirection) — gone from the code (Phase 3 +
  the pool).
- **Cross-cycle fib unification arc** — user sign-off 2026-06-19 (only the bundled follow-up in E remains).
- **Main sid=0 cycle=0 true-first-breakout (Commit 2)** and the **(c) zone-pattern inversion pass** — done
  2026-06-16 / 2026-06-20.
- **Never-established-cycle fallback POI** — resolved 2026-09-29 (option N; PLAN_D §7.3).

---

## Idea Template (for a new idea)

### Idea Name
**Problem:**  
**Proposed Solution:**  
**Integration Point:**  
**Why Deferred / Trigger:**  
Then add its one-line entry to the right bucket above.
