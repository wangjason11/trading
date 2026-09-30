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

## A. Week 8 close-out (next)

**Chart review with the user** (Week 8 DoD: "HTF and LTF context aligns logically; confluence vs non-confluence is
visually obvious")
- **Zone-proximity candle-direction filter** — tried and reverted (the user changed their mind about the approach);
  re-discuss first. DECIDE. Detail: memory `project_main_structure_debug_plan.md` (step 4, the revert).

**Hygiene (a Light session)**
- **§13.5.d leftovers** — the carve-outs are gone (subsumed by the 2026-05-25 lifecycle redesign, LANDMINES "Var 3 +
  Var 4 Last-Per-Cycle Carve-Outs Are a Pair"), but `stash@{0}` ("WIP §13.5.d") remains (drop it — user OK needed),
  two stale docstrings say "carve-out remains until §13.5.d" (`multitf/subsequent_confluence_pipeline.py:19`,
  `multitf/subsequent_counter_pipeline.py:19`), and memory `project_part4_blocker_135d.md` still reads as a live
  blocker. OPEN.
- **Dead code** — `patterns/imbalance.py` `get_unfilled_imbalances` / `has_imbalance_in_range` + the unused import at
  `zones/fib_tracker.py:26` (IMBALANCE_FILL_SEMANTICS: "hygiene follow-up: delete" — but memory lists a `get_unfilled_
  imbalances` direction filter as deferred: resolve that conflict, DECIDE); the legacy `compute_structure_from_start`
  / `compute_structure_scenario_3` / `identify_start_scenario_2_after_reversal` (no production caller; deletion
  deferred until their callers retire — `structure/MARKET_STRUCTURE_SPEC.md` ~:612). OPEN.
- **Stale docs found by the 2026-09-30 sweep** (claims no longer true):
  - proximity pips — GLOSSARY (`proximity_pips`) and ARCHITECTURE ("Threshold defaults") say H1 9 / M15 6 / M5 3,
    PART4 §4.4 (~:818) and memory `project_unified_identify_start_probe.md` (~:143) describe a pending change to
    8/6/4 or M15 6 / M5 4 — the code (`zones/zone_proximity.py` `DEFAULT_PROXIMITY_PIPS`) is **8 / 5 / 3**;
  - "the deferred main sid=0 cycle=0 fix (Commit 2) … not yet wired" — GOTCHAS (~:1637) and MARKET_STRUCTURE_SPEC
    (~:81): it IS wired (`structure_engine.compute_structure` passes `enforce_cts0_new_extreme`);
  - GOTCHAS (~:125) calls the "(c) other-zone-pattern inversion pass" deferred — DONE 2026-06-20 (its own :89);
  - `engine_v2/README_PROJECT_OVERVIEW.md` "In progress / deferred" dates from Weeks 6–7;
  - memory files still saying "deferred" for resolved things: `project_lifecycle_convention_klzone_fibstate.md` (the
    KL `active` / `deactivated_by` indirection is gone from the code), `project_sub_structure_lifecycle_redesign.md`
    "Remaining (both DEFERRED)" (FibState scalar lifecycle done; WVMI done — Plan G), `project_main_structure_debug_
    plan.md` "open blocker before §13.5.d" (Scenario 2 mismatch — resolved by the cross-cycle fib unification),
    `project_part4_progress.md` "Step 3 next — what's deferred". OPEN.
- **Charting audit 2026-09-21** — 16 verified doc-staleness / style items (fold in when those files are touched) and
  two design questions: add an M15 fib renderer or drop the inert `fib: lines` toggle; delete or document the no-op
  M15 dot-opacity tier. OPEN / DECIDE. Detail: memory `project_charting_audit_20260921.md`.
- **Naming-Standard audit of never-audited index keys** — KL `bounds_steps[*].start_idx` (the base ANCHOR on the INIT
  step, a MOMENT on expansion steps — one key, two kinds), RANGE_STARTED `confirm_idx` / `start_idx`, STATE_CHANGED
  `effective_idx`. Parked by the user ("don't start unasked"); cheap; the `bounds_steps` one should land before
  anything reads zone expansions for entries / stops. DECIDE when. Detail: `engine_v2/GLOSSARY.md` "Naming Standard".

## B. Week 8 scope — needs the user's decision

- **The M5 layer** (`H1.main >> M5.counter`) — the Part 4 vision's 4th chart for the strategy (memory
  `project_part4_vision.md`), PART4 §13.6 (recursive depth), and the syllabus's Week 8 "run pipeline on 3 TFs". The
  pool's identity tuple is recursion-ready but M5 nesting is not built (PART4 §17.12). Related open decision: the
  canonical encoding of a deeper path, e.g. `H1.main >> M15.counter >> M5.confluence` (PART4 ~:176 "Open (deferred)").
  A Heavy plan, not a close-out task. DECIDE (in Week 8, or later).
- **Syllabus Week 8 extras** — chart dropdowns (pair / TF), a "show HTF context" checkbox, and a `ContextSnapshot` (HTF
  trend, active HTF zones, LTF candidates); none exist. Part 4's entity / registry design may supersede the snapshot.
  DECIDE (keep or drop). Detail: `engine_v2/SYLLABUS.md` Week 8.

## C. Part 4 closure (when Part 4 is declared done)

- **§13.5.e remainder** — delete the orchestrator's deprecated `s_res.df.attrs[...]` writes; blocked until the H1 chart
  reads its overlays through the registry (the chart-fallback half is DONE). OPEN. Detail: PART4 §13.5.e (~:2270).
- **§13.7 doc finalization** — incl. the §16.10 file-naming deviation (spec `H1.main__M15.confluence.html` vs the code's
  `{H1_basename}_M15_confluence.html`). OPEN. Detail: memory `project_part4_progress.md` (Step 3e note).
- **Delete `engine_v2/PRE_REFACTOR_INVARIANTS.md`** ("will be deleted post-refactor", memory `project_part4_vision.md`)
  and retire memory `project_part4_workflow.md` ("retire when Part 4 is complete"). OPEN.

## D. Before Entries (Week 9 prerequisites)

- **Zone strength scoring** — syllabus Week 7 (`strength_score` / `strength_flags`, "strong zones look strong", click a
  zone → strength + flags); `KLZone.strength` is always `0.0` (`zones/kl_zones_v1.py`). DECIDE (scope). Idea attached:
  scale strength thresholds by ATR percentile (volatility regimes).
- **Single-cycle fib POIs on subs** — let single-cycle fibs coexist with cross-cycle ones (additive). DECIDE. Detail:
  memory `project_sub_single_cycle_fib_pois.md`; `engine_v2/zones/CROSS_CYCLE_FIB_SPEC.md` §13.
- **Pinbar body-pip floor calibration** — provisional values + the reclassify-to-pinbar approach itself; "revisit
  during strategy optimization". TRIGGER (strategy optimization). Detail: memory `project_pinbar_body_pip_floor.md`,
  `engine_v2/features/CANDLE_BODY_FLOOR_NOTES.md`.
- **Should subs get the DERIVED CTS KL zone as their probe reference?** (today every sub reference zone is ad-hoc).
  Unscheduled follow-up. DECIDE. Detail: PART4 §17.12, GOTCHAS (the `kl_zones=[]` pool-path note).

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
- **Cascade-viz hover extras** — `current_versions` vs peak; a V90 → V60 → V30 downgrade cue. Detail: memory
  `project_item_5_cascade_viz.md`.
- **Breadcrumb header with cross-chart navigation** — "not in v1". Detail: PART4 §16.9.

## H. Deferred indefinitely

- `armed_idx` chart surfacing; consumer-specific fill predicates (IMBALANCE_FILL_SEMANTICS "When to revisit"); a
  `get_unfilled_imbalances` direction filter (conflicts with the "delete" hygiene item in A — resolve there).

## I. Needs confirmation (possibly superseded)

- **"(b) the (0,0,1) start-logic update"** — a user-driven follow-up noted 2026-06-01 (memory
  `project_unified_identify_start_probe.md` ~:188); probably absorbed by the true-first-breakout cycle-0 redesign a
  week later (memory `project_true_first_breakout_cycle0.md`). CONFIRM.

---

## Recently closed (history pointers)

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
