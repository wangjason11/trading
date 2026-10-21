# PLAN_S — the imbalance fill rule (65 % arming, separate candles) + IC Step 1B, the closest-prior-imbalance preference (the user's IC revisit, 2026-10-20 → 21)

**Status: steps 1–2 LANDED 2026-10-21** — step 1 source `69c099b`, save **S_S1** `20261021_020427_69c099b` (save
`b4f040b`, trunk `138c2a7`) = §2.2 exactly; step 2 source `cbaed31`, save **S_S2** `20261021_081421_cbaed31` (save
`780e628`, trunk `f4e83c6`) = §3.2 exactly (§6). **Step 3 BUILT 2026-10-21 (a new session):** the fresh audit +
the engineering reading §4.3, the variant measured (§4.4 — 97 cells, no cascade), names + hover decided (§4.3.1, rounds
6–7), the plan cold review + the landing review (§5, §6), `/compare` vs S_S2 = §4.4 exactly; the user's chart look +
go: "looks good go ahead" (§6). Baseline at drafting **S_R3**
`artifacts/commits/week9-entries-decisions/20261020_225749_d424099` (source `d424099`, PLAN_R step 3). HEAD at drafting `896db93`; suite 1839 (green
2026-10-20, 140.7 s). Session order item 4 (register `IDEA_PARKING_LOT.md` §D "SESSION ORDER"). THREE steps, in this
order, each with its own prediction (measured before its build), `/compare` and chart look: §2 the 65 % arming
threshold → §3 the separate-candles fill rule → §4 Step 1B. Commits: the user's call at each `/compare` pause (the
author suggests one per step — 65 % is a global parameter the user may keep without 1B).

Inputs (`qa_ic_revisit_inputs/`): the user's write-up verbatim `user_writeup_20261020.md`; every Q&A round and decision
verbatim + the measurements `qa_findings.md` (rounds 1–5); the read-only shadows `shadow_step1b.py`,
`shadow_fill_rule.py`, `shadow_ic_direction.py` (+ logs); the variant patchers `make_variant_fill65.py` (step 1) and
`make_variant_stroke2_later.py` (step 2, applied on top of step 1's); step 3: the read-only shadow on HEAD
`shadow_step1b_final.py` (+ log — the per-record 1B outcome, candle tables of the moved records, MS's differing
reads) and the behaviour-only variant patcher `make_variant_step1b.py`.

---

## 1. The rules

**Change A — the arming threshold 70 % → 65 %** (write-up, verbatim): "update our parameter that defines what %
threshold price must move past to trigger eventual unfilled imbalance from 70% to 65%." Round 2 (verbatim): "Yes the
70% to 65% change is only on the armed portion of the filled imbalance check. And yes the scope is everywhere."

**Change B — the two fill strokes on separate candles** (round 3, verbatim): "it seems like currently we don't have a
rule that specifies that the same candle cannot meet both rules. That means whatever candle meets the armed condition,
the 2nd condition for filled must be met by a different candle after the armed candle." And: "On edge is ok" (a close
exactly AT the gap's outer edge still counts). Scope: every fill reader (round 2 option (c), chosen in round 3).

**Change C — Step 1B** (write-up, verbatim): "prefer ICs where the closest imbalance prior to IC is still unfilled as of
IC … We also only care about the closest imbalance prior to the IC specifically … a preference condition rather than a
requirement condition … applied after Step 1 … if there are no candidates that meet this preferable condition, then we
simply move on with the original Step 1 candidates to Step 2 … Step 1B would be applied after 1A in the same way".
The write-up's "if only 1 meeting 1B then that will be the final IC" / "At any point if 1B results in just 1
preferred candidate, then that's the final IC" is SUPERSEDED by round 2 (verbatim): "loner goes through Steps 2-3" …
"Actually now thinking more about, it I don't think we need to implement this specifically for loners. In fact, it
should be for all cases when we reach 1B runs and we don't produce a final IC (either fail 1B
immediately, or fail Steps 2-3 after any number of candidates pass 1B). In that case, we always go back and run either
the original Step 1 or original Step 1A candidates (whichever came before the failed 1B) through Steps 2-3." Round 2
Q4: "in direction of structure & opposite of IC. Yes BOS anchor to IC. A gap formed by IC counts … Yes no min size";
Q5 (1B outranks the institutional pattern): "yes". Round 5 (verbatim): "I now want to exclude IC from being able to
fill most recent prior imbalance. Basically the balance needs to be unfilled up to the candle right before the IC."

**Not changed:** IC condition 1 (round 5: "Let's go with a) for condition 1") — the agreed direction (candle +
volume); flat-volume and volume-only-doji candles still qualify (sub 4 cycle 0's IC 3235 is a doji).

---

## 2. Step 1 — the arming threshold 0.70 → 0.65 (change A)

### 2.1 The engineering reading

ONE constant, `IMBALANCE_FILL_THRESHOLD = 0.65`, in `engine_v2/common/types.py` beside `ImbalanceInstance` (the
predicate's home; `common` is imported by every reader, never the reverse). **AUTHOR'S RESOLUTION — confirm the name**
(it keeps the `fill_threshold` parameter family; it is the stroke-1 retrace fraction). It replaces today's six literal
/ duplicated defaults (audit 2026-10-21, `grep "0.70\|FIB_FILL_THRESHOLD"` over production code):

| Site | Today | After |
|---|---|---|
| `common/types.py` `ImbalanceInstance.is_filled(..., fill_threshold=0.70)` | literal | the constant |
| `patterns/imbalance.py` `has_unfilled_imbalance` + `iter_unfilled_imbalances` (`fill_threshold=0.70`, 2×) | literal | the constant |
| `zones/cross_cycle_fib.py` `resolve_cross_cycle_eligibility(..., fill_threshold=0.70)` | literal | the constant |
| `zones/fib_tracker.py` `FIB_FILL_THRESHOLD = 0.70` → `FibTrackerConfig.fill_threshold` | its own constant | RETIRED; `FibTrackerConfig` defaults to the constant |
| `zones/poi_zones.py` `POIConfig.fill_threshold = 0.70` | literal | the constant |

`FIB_FILL_THRESHOLD`'s importers switch to the constant: `pipeline/orchestrator.py` (import + the downstream tracker's
config, `fill_threshold=FIB_FILL_THRESHOLD`), `structure/structure_engine.py` (import + MS's embedded tracker's
config), and the tests `test_plan_i_s5_ms_reads_the_fib.py` (import, :99, :314, :333) and
`test_plan_i_s5_landing_review_pins.py` (:192–195). No alias is kept.

Behaviour reaches every fill reader: FibTracker (downstream + MS's embedded one — fib activation / deactivation, the
cross-cycle own test + dead-cycle walk), IC condition 3 (`immediate_unfilled_imbalance_first_c2_idx`), the POI
activation sweep's fill cache (`_compute_fill_idx_cache`), MS's in-flight IC read (`poi_inner_for_fib`, the tracker's
value), the imbalance export meta (`armed_idx` / `confirmed_fill_idx`). Tests that pass an explicit `0.70` keep it
(they pin mechanics at a stated threshold); a test that relies on the DEFAULT and breaks is examined one by one (§2.4).

### 2.2 Prediction (measured — the variant replay `%TEMP%/v65`, `make_variant_fill65.py` on `git archive HEAD`)

`cmp_save.py` S_R3 vs the variant: **11 real cells, all three figures identical** (traces / shapes 81/236, 153/138,
301/309, xy-equal); 0 structure-event, POI, KL, trigger, sub or WVMI changes:
- H1 `imbalance_instances.csv` meta: 7 instances arm earlier (275 → 267, 445 → 444, 494 → 493, 510 → 509, 868 → 866,
  1042 → 1041; instance #92 armed + committed 761 → 600);
- sub 3 cycle 0's fib (both lenses' `fib_lifecycle.csv` meta): one more deactivate / reactivate (the meta's last pair
  2667 / 2672 → 2722 / 2723);
- one IC check (both lenses' `ic_candidates.csv`): its immediate gap 3488 fills at 0.65 → fails IC condition 3 (not a
  pick before or after).

The landed replay must equal the variant's outputs (CSV md5 + figure JSON) — the variant patched exactly these
defaults. Fetch gate "Fetched 4228 M15 candles … in 5 chunks"; skip grep 23. Anything else → stop, name the
mechanism, ask.

### 2.3 Docs (same commit)

`IMBALANCE_FILL_SEMANTICS.md` (:44 the c3 example's `0.70·size`, :120 / :175 the signatures, :155–158 "Why two
strokes"; + a dated line naming the change; the "Old vs new" history table :338–346 stays history),
`zones/POI_ZONES_SPEC.md` (:114–124, :333–334), `GLOSSARY.md` "fill check" (:200), `ARCHITECTURE.md` (:295),
`zones/CROSS_CYCLE_FIB_SPEC.md` (:248, :268 — `FIB_FILL_THRESHOLD` retired), code comments (`poi_zones.py` :107,
:962, :1500, :1721; `fib_tracker.py` :125–136 → the constant's own comment in `types.py`). Every rewritten number is
re-checked against the code (memory `feedback_implement_against_docs`).

### 2.4 Tests and mutants

- The pin `test_plan_i_s5_ms_reads_the_fib.py::test_one_fill_threshold` → `IMBALANCE_FILL_THRESHOLD == 0.65 ==
  FibTrackerConfig().fill_threshold == POIConfig().fill_threshold`, plus the four function defaults read with
  `inspect.signature` (`is_filled`, `has_unfilled_imbalance`, `iter_unfilled_imbalances`,
  `resolve_cross_cycle_eligibility`) — so every site is pinned to the one constant.
- Mutants (each must fail the suite): each of the six sites set back to a literal `0.70` (6), the constant set to
  `0.70` (1), the orchestrator's and structure_engine's tracker configs passed a literal `0.70` (2) → 9.
- Suite: 1839 + the pin's new asserts; any default-reliant fixture that breaks is listed in the build log with its
  resolution (an explicit `0.70` only where the fixture's geometry was designed at 0.70 and the test is about
  mechanics, not the value).

### 2.5 Chart look

The figures are identical (predicted) — the chart look is a confirmation that nothing visual moved.

---

## 3. Step 2 — the two strokes on separate candles (change B)

### 3.1 The engineering reading

Stroke 2 counts only on a candle STRICTLY AFTER the arming candle; a close AT the outer edge still counts (`>=` top /
`<=` bottom, unchanged). The rule lives in THREE implementations (audit 2026-10-21, `grep "gap_size \*"`):
1. `ImbalanceInstance.is_filled`'s positional fast path (`common/types.py`, the bull and bear loops) — check stroke 2
   BEFORE latching stroke 1 on a candle (so the arming candle's own close is never tested);
2. its label-based fallback (same file, the bull and bear loops) — the same reorder;
3. `_compute_fill_idx_cache` (`zones/poi_zones.py`, the POI activation sweep's numpy scan) — the stroke-2 slice starts
   at `armed_pos + 1` (bull and bear); its comment "same-candle fill is legal" goes.
Docstrings: `types.py` `is_filled` ("Both can fire on the same candle (rare)"), `poi_zones.py`
`_compute_fill_idx_cache` ("Both can be equal (same-candle stroke 1+2 — rare but legal)").

Measured (`shadow_fill_rule.py` census): same-candle commits are NOT rare — at 0.70, 48 / 152 H1 and 153 / 522 M15
committed fills; at 0.65, 47 H1 and 158 M15 (of 152 / 522). A pinbar whose wick reaches the arming level while its body stays outside the gap
did both strokes alone; it now waits for the next candle that closes outside the gap — whatever its colour (H1 450,
a red candle closing above gap 447's top, completes the fill that 449 armed).

### 3.2 Prediction (measured — the variant replay `%TEMP%/vS` = step 1's patch + `make_variant_stroke2_later.py`)

`cmp_save.py` vs step 1's variant outputs (`%TEMP%/b65`): **51 real cells; 0 structure-event, fib, IC, KL, trigger,
sub or WVMI changes**:
- H1 `imbalance_instances.csv` meta: **47** instances' `confirmed_fill_idx` move later (e.g. 19 → 20, 42 → 47,
  106 → 110, 954 → 964, 992 → 999); row 35 (gap c2 180, armed 183) never commits;
- H1 `poi_zones.csv` row 3 — sid 1 cycle 2 (IC 859): `confirmed_idx` 998 → 954, the activation history loses the
  992 deactivate / 998 reactivate pair (gap 987's fill moves 992 → 999, when gap 997 already keeps the POI active);
- M15 confluence `poi_zones.csv`: sub 1 cycle 2 deactivates 2192 → 2193; sub 4 cycle 0 deactivates 3377 → 3378;
- figures: shapes **236 → 234, 138 → 136, 309 → 307**, traces xy-equal — the H1 POI (.57786–.57850) becomes ONE zone
  2026-01-13 16:00 → 01-19 23:00 instead of two segments + a reactivation line (H1 chart, and the H1 overlay on both
  M15 charts); the two M15 sub POI segments end one candle (15 min) later.
Variant timing: 56.2 s wall (data_fetch 6.24, multi_tf_dual 11.09, chart_h1 6.15, chart_m15_counter 9.30,
chart_m15_confluence 22.27); fetch gate OK; skip grep 23. The landed replay must equal the variant's outputs.

### 3.3 Tests, mutants, docs

- `tests/test_imbalance.py`'s same-candle case flips: the arming candle that also closes past the edge → NOT filled
  at that candle; filled at the next candle that closes at / past the edge (bull + bear; a close exactly AT the edge
  included). The same on the fallback path (a non-RangeIndex frame) and on `_compute_fill_idx_cache` (its
  `(armed_idx, confirmed_fill_idx)` pair: `confirmed > armed`), plus an equivalence pin: the cache's
  `confirmed_fill_idx` == the first `t` with `is_filled(df, t)` on the same fixtures.
- Mutants: each of the six sites (fast bull / bear, fallback bull / bear, cache bull / bear) reverted to the same-candle
  form individually → 6, each must fail.
- Docs: `IMBALANCE_FILL_SEMANTICS.md` "The predicate" (the stroke-2 row `[armed_idx + 1, check_to_idx]`, the
  same-candle sentence → its opposite + why, the edge-case table, "Why two strokes" — the user's round-3 words),
  `POI_ZONES_SPEC.md` §1's fill check, `GLOSSARY.md` "fill check", `ARCHITECTURE.md` :295.

### 3.4 Chart look

The H1 POI sid 1 cycle 2 (IC 859) on the H1 chart (01-13 → 01-19), and the two M15 confluence sub POIs' end edges.

---

## 4. Step 3 — Step 1B (change C)

### 4.1 The rule (the final spec, `qa_findings.md` round 5)

After Step 1 (or, when Step 1 has no candidate, after Step 1A), for each IC candidate: its **closest prior
imbalance** = the sd-direction imbalance instance whose gap-candle (c2) run starts at or after the fib's BOS anchor
and ends before the IC, the one ending LATEST (engine instances: consecutive gap candles merge; no minimum size; a gap
whose c3 is the IC counts; a gap straddling the BOS anchor does not — 0 on the window). The candidate **passes 1B** when
that gap is NOT filled through the candle right before the IC (`is_filled(df, IC − 1)` False — a gap whose c3 is the
IC is unfilled by construction). A candidate with no closest prior gap fails 1B (author's default, round 5's final spec — changes
nothing on the window).

Flow: passers (any number) → Steps 2–3; they give no IC (only possible when every passer is a pinbar with no pattern)
→ Steps 2–3 on ALL the candidates of the step before 1B (Step 1's, or Step 1A's) — which is exactly today's pick; no
passer → Steps 2–3 on all candidates (today's pick). Consequence: **1B never adds or removes an IC; it can only swap
today's IC for a passer that wins Steps 2–3 among the passers.** 1B outranks the institutional pattern. Step 4 stays
deferred. Both consumers read it (one selector, `select_ic`): the drawn POI and MS's in-flight IC read
(`poi_inner_for_fib` → CTS_CONFIRMED by sd-proximity) — and through the sub POIs, the proximity overlay's capped views.

### 4.2 Prediction so far (first order — `shadow_fill_rule.py` SX, at 0.65 + separate candles, IC side; valid because
neither fill change moves a fib, an IC check or a pick)

**Two drawn picks move: sub 0 cycle 2 1216 → 1142; sub 3 cycle 1 2808 → 2821** (2808 = pattern 3B.iii; 2821 the lone
passer, a non-pinbar). Sub 3 cross 1→2 keeps 2930 (pattern 3C.iv — its gap 2926 is not even armed before 2930);
2406 and 2721 pass but are lone no-pattern pinbars → the fallback keeps today's no IC; no-IC stays 7 / 27.

### 4.3 The engineering reading (written 2026-10-21 on HEAD `1765499` = steps 1–2 landed, after a fresh audit)

**The audit.** ONE selector, `poi_zones.select_ic`, serves every consumer — Step 1B is one edit there:
- the drawn POI and the debug CSV: `select_ic_records` (`select_ic(..., classify_all=True)` with the record's FINAL IC
  reference) → `derive_poi_zones` / `derive_ic_candidates`, in every downstream run — the H1 main, every sub
  projection, the proximity overlay's capped views (`orchestrator` step 8);
- MS's in-flight read: `poi_inner_for_fib` (`select_ic` WITHOUT `classify_all`, on the embedded tracker's
  representative fib at each CTS_ESTABLISHED / CTS_UPDATED moment; the frame = `MarketStructure._resolver_df()`, cut at
  the run's data edge, `attrs["imbalances"]` full-frame — Plan A's residual) → CTS_CONFIRMED by sd-proximity;
- the fill predicate `ImbalanceInstance.is_filled(df, check_to_idx, fill_threshold)` scans `(end_idx, check_to_idx]`,
  stroke 2 strictly after the arming candle (step 2); a degenerate gap (`gap_size <= 0`) counts as filled;
- the instances `df.attrs["imbalances"]` (`compute_imbalance`): consecutive same-direction c2s merged, `start_idx ..
  end_idx` = the c2 run, the bounds from its first c1 / last c3.

**Every bound, traced** (memory `feedback_trace_every_bound_before_claiming`):
- lower: `start_idx >= fib_state.bos_idx` — the record's own BOS anchor (a cross's too; the same bound as Step 1's
  window). A run starting AT the BOS anchor counts; one straddling it does not (`start_idx < bos_idx <= end_idx` — 0
  on the window, `shadow_step1b_final.py`);
- upper: `end_idx < ic_idx`. An sd run cannot contain the IC (a c2's candle direction is sd, a checked candle's −sd or
  0 — the structure `immediate_unfilled_imbalance_first_c2_idx` asserts), so `end_idx <= IC − 1`: a run whose c3 IS
  the IC (`end_idx = IC − 1`) counts (round 2 Q4);
- "closest" = the largest `end_idx` (instances never overlap → unique);
- **provenance** (plan cold review lens A, 2026-10-21): the user's own words fix the direction ("in direction of
  structure & opposite of IC"), the window ("BOS anchor to IC"), "A gap formed by IC counts" and "no min size". Two
  details are the AUTHOR'S readings — reading P of round 1 (`qa_findings.md` "Primary reading P", sent to the user in
  round 1; round 2 accepted its window, not these details explicitly): (a) a gap = an ENGINE instance (consecutive sd
  c2s merged, the fill tested on the merged bounds first c1 → last c3) — it decides on the window: both of sub 3
  cycle 1's deciding gaps are merged runs (2759..2760, 2810..2812); (b) a run straddling the BOS anchor does not
  count — 0 on the window. Like "no prior gap → fails" (round 5's final spec), both are the author's defaults the user can revisit
  (all accepted at the chart look, 2026-10-21 — §6 Step 3);
- the fill horizon: `is_filled(df, IC − 1, config.fill_threshold)` scans `(end_idx, IC − 1]` — empty for the gap whose
  c3 is the IC (unfilled by construction — EXCEPT a degenerate merged gap, `gap_size <= 0`, which `is_filled` calls
  filled before scanning, `types.py` — 0 on the window); the IC's own candle is never read (round 5: "unfilled up to
  the candle right before the IC");
- knowability: every candle read is ≤ the IC ≤ the fib's CTS anchor ≤ the reader's moment (the drawn record: the
  structure's end; MS: the fed event's moment, its frame cut at the data edge). No look-ahead; no `evaluated_at` (IC
  condition 3's class: retrospective on the fib). The full-frame instance list cannot leak: an instance ending before
  the IC has its last c3 ≤ the IC;
- the threshold = `config.fill_threshold` (the drawn POI's `POIConfig()` and MS's tracker value — the one constant,
  pinned).

**The code** (`zones/poi_zones.py`; names PROPOSED — §4.3.1):
1. `closest_prior_imbalance(df, ic_idx, fib_state) -> Optional[ImbalanceInstance]` — the rule above.
2. `ICCandleCheck` + two fields, default None = not evaluated: the gap's first c2 (None = no prior gap) and its
   unfilled flag (True = passes 1B; False = filled through IC − 1, or no prior gap). Evaluated in `select_ic` for the
   IC candidates and, with `classify_all`, for EVERY check (the drawn record → the CSV, as the pattern is) — MS's read
   pays only for its candidates. `is_ic_candidate` is unchanged: 1B is a preference, not an IC condition.
3. `select_ic` after Step 1 / Step 1A: passers = the IC candidates passing 1B → Steps 2–3 (today's code, factored into
   one private helper over a candidate list); no IC from them → Steps 2–3 on ALL the candidates; no passer → all the
   candidates. The selection's Step 2 / Step 3 fields (`tiebreak`, `ip_outcome`, `ranking`, `no_ic_reason`) are those
   of the run that gave the answer (over the passers, `ranking_in_play_count` counts passers).
4. `ICSelection` + the 1B outcome (a REQUIRED field — every construction states it): the IC came from the passers /
   passers gave no IC → all / no passer → all / None (no IC candidate — 1B never ran). The two counts are PROPERTIES
   derived from `checks` (the candidates; the candidates passing), None with the outcome — one source, no stored copy
   (lens B). Asserted at construction (strengthened by lens B, 2026-10-21): **the pick is an IC candidate** (`pick is
   None or pick.ic_idx ∈ ic_candidates` — missing today: `checks=()` with a pick was accepted; a Steps-2–3 helper fed
   `checks` instead of the candidates would silently pick a non-candidate on the drawn path and crash on MS's);
   None ⇔ `ip_outcome` None; outcome not None ⇒ every candidate evaluated; `preferred` ⇒ a pick that passes;
   `fallback_passers_no_ic` ⇒ ≥ 1 passer and the pick None or a NON-passer (every passer is a no-pattern pinbar);
   `fallback_no_passer` ⇒ no candidate passes; a check with `prior_imbalance_unfilled` True has a gap with first c2
   `< ic_idx` (AS BUILT — the selection has no fib, so the `bos_idx <=` lower bound is enforced by
   `closest_prior_imbalance` itself, pinned by its tests + the `gap_bos_*` mutants; landing review lens 1). `derive_ic_candidates` asserts every check row evaluated (like its `pattern`
   assert). "1B never adds or removes an IC" is NOT asserted (it needs a second Steps-2–3 run — double MS's cost);
   it is TESTED (fixtures: the pick is None ⇔ the all-candidates run's pick is None) and the shadow asserts it on 27 +
   105 records; the `fallback_passers_no_ic` assert is its cheap structural half.
5. The POI meta: the outcome + the IC's own prior gap's first c2 (`_ZONE_META_IDX_KEYS` — the sub mirror shifts it;
   `tests/test_event_meta_idx_keys.py` lists poi_zones' index keys) + whatever the hover needs (§4.3.1).
6. The CSV (`ICCandidateRecord`, `export_ic_candidates`, `IC_CANDIDATE_COLUMNS`; the `_idx` in
   `_IC_CANDIDATE_IDX_FIELDS` — `test_mirror_shifts_every_index_field_once_per_lens` asserts the set): the two per-check
   values on every check row (empty on the no-IC row), the outcome on every row of the record (the no-IC row
   included, like `ip_outcome`) — five columns in all with the two counts (§4.3.1), placed in step order between
   `ic_zone_band` (Step 1A) and `ip_outcome` (Step 2): `prior_imbalance_first_c2_idx`, `prior_imbalance_unfilled`,
   `prior_imbalance_outcome`, `prior_imbalance_passer_count`, `ic_candidate_count`. **A save-format boundary:** a
   save before step 3 lacks the five columns and the six POI meta keys → `/compare` keyed by column name
   (`qa_ic_crossfib_inputs/cmp_trial.py`).
7. The hovers (`charting/_zone_render.py`) — §4.3.1. Measured by the variant: with today's hover code the two moved
   POIs would read `No pattern · only IC candidate` (POI hover) / `No pattern · IC 1142 · only IC candidate` (fib
   hover) — WRONG: 1142 is the only PASSER of 3 candidates, 2821 of 2 (ICR-10's `ranking` None = one candidate
   entered Step 3). The wording must name the preference.

#### 4.3.1 Names and hover — DECIDED by the user 2026-10-21 (round 6, `qa_findings.md`)

The user chose (verbatim, the two selected options): **"prior_imbalance_* (Recommended)"** and **"Fix + narrowing line
(Recommended)"**. Every candidate offered was listed in text first: gap index `prior_imbalance_first_c2_idx` /
`closest_prior_imbalance_first_c2_idx` / `preceding_imbalance_first_c2_idx`; pass flag `prior_imbalance_unfilled` /
`passes_prior_imbalance_preference` / `has_unfilled_prior_imbalance`; outcome `prior_imbalance_outcome` (preferred /
fallback_passers_no_ic / fallback_no_passer) / `prior_imbalance_preference` (applied / no_ic_from_passers / no_passer)
/ `step1b_outcome`. The rule's name in docs / code: **the prior-imbalance preference** (the user's "Step 1B").

| Where | Name | Value |
|---|---|---|
| `ICCandleCheck` + CSV, every check row | `prior_imbalance_first_c2_idx` | the closest prior sd gap's first c2; empty = no prior gap (or the no-IC row) |
| `ICCandleCheck` + CSV, every check row | `prior_imbalance_unfilled` | True = passes 1B; False = filled through IC − 1 or no prior gap; empty on the no-IC row |
| `ICSelection` + CSV every row of the record + POI meta | `prior_imbalance_outcome` | `preferred` (the IC came from the passers) / `fallback_passers_no_ic` (passers, no IC from them → all the candidates) / `fallback_no_passer` (no passer → all); None = no IC candidate (1B never ran) |
| `ICSelection` + CSV every row of the record + POI meta | `prior_imbalance_passer_count` | the passers; None with the outcome |
| `ICSelection` + CSV every row of the record + POI meta | `ic_candidate_count` | the IC candidates (Step 1's or Step 1A's); None with the outcome |
| POI meta | `prior_imbalance_first_c2_idx` | the IC's own closest prior gap (`_ZONE_META_IDX_KEYS`) |

The two counts are the hover option's "2 count keys"; they also go on the CSV rows because the fib hover is built
from the CSV's picked row (`ic_rows_by_fib_record`) — the two hovers read one value, nothing re-derived. **The
AUTHOR'S defaults inside the decision** (lens A): the two count NAMES (`prior_imbalance_passer_count`,
`ic_candidate_count`) were not among the offered names — accepted at the chart look (2026-10-21, §6 Step 3); "nothing when it fell back"
WAS in the selected option's text (recorded verbatim in round 6), so a pick that itself fails the preference shows
no 1B line — on the window the five fallback records with an IC: 650, 923, 1934, 3809 (no passer) and 2575 (its
lone passer a no-pattern pinbar → the fallback).

**Round 7 (the user, 2026-10-21 — after the plan review's lens A MAJOR; `qa_findings.md` round 7 verbatim): "Name the
outranked pattern".** (1) `ip_outcome` = Step 2's outcome over the candidates Steps 2–3 RAN ON (the passers when the
preference applied) — POI_ZONES_SPEC + GLOSSARY redefine it; on sub 3 cycle 1 it reads `no_pattern` although the
non-passer 2808 carries 3B.iii. (2) Two more POI meta keys (POI hover only — not the CSV, not the fib hover):
`prior_imbalance_outranked_ic_idx` (`_ZONE_META_IDX_KEYS`) and `prior_imbalance_outranked_ip_rank` — the IC Step 2
would pick over ALL the candidates (best (rank, shape) + the tiebreak rules) when that IC FAILS 1B, else None (the
author's reading of "the best pattern the preference outranked"; computed only when the preference narrowed —
otherwise provably None). `ICSelection` field `prior_imbalance_outranked` (an `IPResult` or None; asserted: set ⇒
outcome `preferred`, a ranked pattern, a candidate failing 1B). On the window: only sub 3 cycle 1 (2808, 3B.iii); sub
8 cross 0→2's all-candidates pick 4131 is itself a passer → None. The names follow the family — author's default,
accepted at the chart look (2026-10-21, §6 Step 3).

**The hovers.** The preference NARROWED the field ⇔ `prior_imbalance_outcome == "preferred"` and passers < candidates.
Then: the POI hover adds `Prior imbalance <c2> unfilled · preferred <passers> of <candidates> IC candidates` (after
the IP / ranking line, before the 80–100 band line); a ranked IC with no ranking rule (one candidate entered Step 3)
reads `No pattern · only preferred IC candidate` (POI hover) / `No pattern · IC <idx> · only preferred IC candidate`
(fib hover) instead of `… only IC candidate`. Nothing changes when every candidate passed or 1B fell back (e.g. sub 3
cross 1→2's 2930: `IP 3C.iv · legs …`, no 1B line). The narrowing line ends ` (over pattern <rank> at <IC>)` when an
outranked pattern exists (round 7). On the window: sub 0 cycle 2 — `No pattern · only preferred IC candidate` +
`Prior imbalance 1140 unfilled · preferred 1 of 3 IC candidates`; sub 3 cycle 1 — `No pattern · only preferred IC
candidate` + `Prior imbalance 2810 unfilled · preferred 1 of 2 IC candidates (over pattern 3B.iii at 2808)`; sub 8
cross 0→2 — its `IP 3B.iii …` line + `Prior imbalance <c2> unfilled · preferred 1 of 2 IC candidates`.

### 4.4 Prediction (measured — the variant replay `%TEMP%/v1b`, `make_variant_step1b.py` on `git archive HEAD`, behaviour only)

`cmp_save.py` S_S2 vs the variant: **97 real cells, all in the M15 lenses' `ic_candidates.csv` / `poi_zones.csv`; 0
structure-event, fib, KL, imbalance, trigger, sub or WVMI changes; the H1 chart identical** (figure + hovers):
- confluence sub 0 cycle 2 (fib 1107..1245, buy): IC **1216 → 1142** — `ic_candidates` 5 cells (1142 picked; 1216
  unpicked, its `ranking_rule` `ic_candle_type` / runner-up 1142 / in-play 3 cleared); `poi_zones` row 4 (6 cells):
  start 2025-12-03 14:00 → 12-02 19:30, top / bottom .57478 / .57346 → .57326 / .57301, `confirmed_idx` 1230 → 1224
  (the CTS-established floor), meta (ranking cleared);
- sub 3 cycle 1 (fib 2758..2842, sell), BOTH lenses: IC **2808 (pattern 3B.iii) → 2821 (ranked, the lone passer)** —
  `ic_candidates` 36 cells per lens (`ip_outcome` pattern → no_pattern on all 34 rows of the record + the two
  `picked` flips); `poi_zones` 7 cells (start 12-29 04:00 → 07:15, top / bottom .58318 / .58279 → .58233 / .58182,
  `ip_rank` 3B.iii → empty, `confirmed_idx` 2829 → **2841**, meta). 2841, not 2829: the activation's touch re-check —
  2821's high .58233 reaches the as-of 61.8–80 zone of the fib (BOS .58424) only once the CTS extends to .57907 (the
  CTS_UPDATED at 2841: zone low .58227; at .57966, 2840, it is .58249);
- figures: H1 identical; M15 confluence traces / shapes 301 / 307, counter 153 / 136 (counts unchanged), the moved
  POI traces' x (4 confluence, 2 counter); hovers (`cmp_hover.py`): 0 templates, customdata of 6 confluence / 3
  counter traces — the two POIs' lines + the fib hovers' IC outcome (`cmp_hover_strings.py`);
- the cascade is empty: MS's in-flight reads whose POI inner differs (first order, `shadow_step1b_final.py`): **14 /
  464**, all in the two moved cycles' geometry runs (sub 0 cycle 2 at moments 1224–1245, sub 3 cycle 1 at 2841–2842) —
  both cycles confirm their CTS by the pullback path (`one_maru_continuous`, 1257 / 2844), so no event moves; the
  proximity overlay's capped views: 12 / 105 picks move, 0 triggers.
- Unchanged, as predicted (§4.2): sub 3 cross 1→2 keeps 2930; no-IC stays 7 / 27; 1B outcome counts on the 27 drawn
  records — from the passers 15, passers gave no IC 5, no passer 5, no IC candidate 2; 29 of 43 candidates pass.
- Variant timing: 57.5 s wall (data_fetch 7.27, multi_tf_dual 10.88, chart_h1 6.06, chart_m15_counter 9.18,
  chart_m15_confluence 21.98); fetch gate PASS (4228 / 5 chunks); skip grep 23.

The landed replay must equal the variant on every existing column (keyed compare: landed vs the variant = only the
new columns / meta keys / hover lines), and vs S_S2 = these cells + the new ones. **The landed build's additions
(predicted from §4.3.1 + the shadow):** the 5 new `ic_candidates.csv` columns on every lens (a save-format boundary);
the 6 new keys in every POI's meta (4 + round 7's 2); the narrowing hover line on exactly THREE POI RECORDS (sub 3 cycle 1's renders on
both lens charts) — sub 0 cycle 2 (1142, 1 of 3), sub
3 cycle 1 (2821, 1 of 2) and sub 8 cross 0→2 (4131, pattern 3B.iii, 1 of 2 — its pick does not move) — and `only
preferred IC candidate` on the first two (POI + fib hovers); every other record's preference either passed all its
candidates or fell back (no hover change).

### 4.5 Tests and mutants (planned)

- **The existing suite under 1B (measured: the suite run inside the variant `%TEMP%/v1b` = 34 failed / 1834 passed,
  in 6 files — `test_institutional_pattern` 9, `test_plan_l_landing_2` 14, `test_plan_k_landing_b` 3,
  `test_plan_p_ic_condition3_step1a` 5, `test_plan_h_landing_b` 2, `test_plan_k_landing_a` 1).** One cause: the
  fixtures' synthetic never-filled gaps (each candidate's immediate gap, IC condition 3) are ALSO the next
  candidate's closest prior gap, while the oldest candidate has none — so 1B narrows the candidates and reshapes the
  picks those tests pin for Steps 2–3 (e.g. `two_immediate`: 7 has no prior gap, 13's is 8 → 13 alone instead of rule
  3's 7). Resolution: those fixtures become 1B-NEUTRAL — a never-filled sd gap at the BOS anchor (`_far_gap(bos)`; it
  touches no other IC read: condition 3, the retired condition 3, the most-recent-before-imbalance reference and the
  POI sweep all read gaps AFTER a candidate ≥ the BOS anchor) so every candidate passes and Steps 2–3 run over all of
  them, as designed — with an assert that 1B was neutral (passers == candidates) where the selection is visible. A
  fixture whose IC is the BOS anchor candle itself (Step 1A) cannot be neutralised that way (no gap can end before
  it) — handled one by one, each listed in the build log.

- `closest_prior_imbalance`: the latest-ending sd gap; a −sd gap ignored even when closer; a run starting AT the BOS
  anchor counts, a straddling one does not; a gap whose c3 is the IC counts; a merged run = one gap (its end).
- Pass / fail: filled through IC − 1 (armed, then committed by a later candle before the IC) → fails; armed before the
  IC and committed BY the IC's close → passes (the round-5 rule); armed only → passes; no prior gap → fails, first c2
  None; a degenerate gap → fails (`is_filled`'s safe default; 0 on the window).
- The flow through `select_ic` (fixtures with ≥ 2 IC candidates): a passer beats a non-passer today's rules would pick
  (outcome "from the passers"); 1B outranks the pattern (a non-passer with a pattern vs a no-pattern passer → the
  passer); every passer a no-pattern pinbar → today's pick (passers gave no IC); no passer → today's pick; no IC
  candidate → None; 1B after Step 1A. Each test proves it ran its path (the outcome value asserted).
- `ICSelection`'s new asserts; the CSV columns (every row / every record row / empty on the no-IC row); the mirror
  shift of the new `_idx` (CSV + POI meta); the hover lines.
- Added by lens B: `select_ic(classify_all=False)` (MS's path — the pick moved, a non-candidate's 1B fields None);
  the hover's two negative cases (every candidate passed → no line; passers < candidates but a fallback, e.g. 1 of 2
  with the passer a no-pattern pinbar → no line); the Step 1A test also asserts `ic_zone_band`; a fixture with a
  NON-default `POIConfig(fill_threshold=…)` (else dropping the threshold argument is an equivalent mutant under the
  one constant); the "filled through IC − 1" fixture commits EXACTLY at IC − 1 (kills a horizon of IC − 2).
- Blast radius (lens B; each fails loudly, listed for the build): `test_event_meta_idx_keys.py` — the `must_see`
  set (:321) + `prior_imbalance_first_c2_idx`, `NON_INDEX_INT_META_KEYS` (:381) + the two counts; positional
  column asserts `test_plan_l_landing_2.py` :351–352, `test_plan_k_landing_a.py` :561; hand-built hover metas /
  rows `test_plan_p_ic_condition3_step1a.py` :215, `test_plan_l_landing_2.py` :384 (`_ranked_meta`),
  `test_fib_chart_render.py` :45 (direct `meta[...]` reads KeyError without the new keys); the `ICSelection` `ok`
  fixture (`test_plan_l_landing_2.py` :193 — a pick with `checks=()` fails the new pick-is-a-candidate assert).
- Mutants (`qa_ic_revisit_inputs/mutants_plan_s_step3.py`, each must fail the suite): the BOS bound `>=` → `>` and
  dropped; the c3-is-the-IC gap excluded (`end_idx < IC − 1`); "closest" = the earliest; the direction filter
  dropped; the fill horizon IC − 1 → IC and → IC − 2; the threshold argument dropped; the Steps-2–3 helper fed every
  check instead of the candidates (passers ⊄ candidates); the POI meta's gap taken from the wrong check; the Step 2 /
  3 fields taken from the passers' run in the fallback; no prior gap → passes; the fallback when the passers give no IC dropped;
  the passers ignored; each outcome value swapped; `classify_all` not evaluating every check; the CSV column / POI
  meta key missing; the mirror lists missing the new `_idx`; the counts (passers ↔ candidates swapped, off by one);
  the hover's narrowing test (`<` → `<=`, the outcome test dropped) and its wording, on the POI and the fib hover.
  NOT mutants (equivalent — lens B): "closest" by the largest `start_idx` (instances never overlap) and `end_idx <=
  ic_idx` (the IC can never be an sd c2: its candle direction is −sd or 0, a c2's is sd).

### 4.6 Docs (same commit)

Widened by the plan cold review (lens A, 2026-10-21 — each line re-checked against the code at edit time):
- `POI_ZONES_SPEC.md` §3: the step list (:368 "among the IC candidates", :372 "the UNFILTERED Step 1 list", :429
  Step 1A's "go through Steps 2–3 as usual") + a new "Step 1B — the prior-imbalance preference" section; §3.5 the
  five columns; §3.6 — each line that assumes Steps 2–3 run over ALL the candidates (:640 "over the UNFILTERED IC
  candidates", :645 "none left → no IC" — over the passers it falls back, :663 the one-candidate empty-rule case,
  :672 `ranking_in_play_count` "rule 1: the Step 1 count"); §4 "Zone Data Fields" (6 keys); §5 "Hover Data". Stays
  true (checked): §3.4 :558–562 and §3.6 :654–656 — the passers are a subset that passed condition 3.
- `charting/CHARTING_SPEC.md` §7 (the POI hover, :150–158 "`No pattern · only IC candidate`") and §8 (the fib hover).
- `GLOSSARY.md` :179 (the CSV), :181 (the ranking rules — "UNFILTERED", the one-candidate case), :183 (`ip_outcome` —
  see the open question on its meaning), + the rule's name.
- The pick's uncut imbalance reads — 1B adds one: `structure/MARKET_STRUCTURE_SPEC.md` :446–451, `LANDMINES.md`
  :1874 and :2482; `IMBALANCE_FILL_SEMANTICS.md` (1B as a fill reader: through IC − 1).
- `ARCHITECTURE.md` :316–320 (the IC steps); `GOTCHAS.md` :2061, :2079 (the IC-candidate check recipe).
- Code docstrings: `poi_zones.py` module (:21), `apply_ranking_rules` (:620–636), `ICSelection` (:664–675),
  `select_ic` (:715–730).
- The register §D item 4; this plan's §6; the `/compare` skill's entry after the save; memory.

### 4.7 Chart look

The confluence chart: sub 0 cycle 2's POI moves from 1216 (12-03 14:00, .57346–.57478) to 1142 (12-02 19:30,
.57301–.57326), active from 1224; sub 3 cycle 1's POI (both lens charts) from 2808 (12-29 04:00) to 2821 (07:15,
.58182–.58233), active from 2841; the hovers' new wording.

---

## 5. Reviews and cost

- Step 1: no multi-agent review — a constant, measured exactly by its variant; my 9 mutants.
- Step 2: a landing review, 1 combined conformance + mutation lens (≈0.2M tokens) — three implementations of one rule
  and a flipped pinned behaviour.
- Step 3: a plan cold review (≈0.6M) and a landing review (2 lenses, ≈0.5M).
- Total ≈1.3M subagent tokens, ≤ 6 agents — under the ~2M / 10-agent ask threshold; each is announced with its size
  before it runs.
- **Step 3's plan cold review — DONE 2026-10-21: 2 read-only lenses, 380,363 tokens by the harness** (A conformance
  191,176; B engineering + measurement 189,187). No BLOCKER. A: 1 MAJOR (the meaning of `ip_outcome` / "No pattern"
  on sub 3 cycle 1 — the pick moves off a PATTERN candidate; put to the user), 4 MINOR (the docs list → §4.6; the
  column count; the provenance of the merged-gap and straddle readings → §4.3; the author's defaults inside round 6
  → §4.3.1), 3 NIT (the degenerate-gap qualifier; the write-up's "loner" superseded → §1; the two "fallback" words).
  B: §4.4 REPRODUCED exactly (the shadow line for line; the variant 97 cells, hovers 6 / 3); 1 MAJOR (no assert that
  the pick is an IC candidate → §4.3 item 4), MINOR (stronger asserts, the counts as properties; the blast radius;
  surviving / equivalent mutants and missing tests → §4.5), NIT (the shadow's `[MS]` lines labelled "H1main" are
  sub geometry runs with slice-local indices — the label fixed in the script). Measured cost of 1B: 0.10 s over
  8731 checks (132 `classify_all` calls) + 0.01 s over 583 candidates (453 MS reads) — < 1 % of the pipeline.

---

## 6. Build log

### Step 1 — LANDED 2026-10-21 (`69c099b`, save S_S1 `20261021_020427_69c099b`; the user: "go ahead, commit step 1
and move on to step 2")

- Code: `IMBALANCE_FILL_THRESHOLD = 0.65` in `common/types.py`; the defaults of `is_filled`,
  `has_unfilled_imbalance`, `iter_unfilled_imbalances`, `resolve_cross_cycle_eligibility`, `FibTrackerConfig`,
  `POIConfig` read it; `FIB_FILL_THRESHOLD` retired (`orchestrator.py`, `structure_engine.py` and the two S5 test files
  import the constant). The pin `test_one_fill_threshold` = the constant == 0.65 == both configs + the four function
  defaults (`inspect.signature`).
- Suite: **1839 passed** (69.4 s) — no fixture relied on the old default (every threshold-sensitive fixture passes an
  explicit `0.70`).
- Mutants (`qa_ic_revisit_inputs/mutants_plan_s_step1.py`): **9 / 9 KILLED** — the 7 value sites by the pin alone
  (no behaviour test distinguishes 0.65 from 0.70: the fixtures pass explicit thresholds; the pin IS the value guard),
  the downstream tracker's wiring by `test_the_downstream_tracker_uses_the_one_config`, MS's embedded tracker's by 3
  tests.
- Replay: 54.3 s wall (data_fetch 2.84, multi_tf_dual 11.66, chart_h1 6.32, chart_m15_counter 9.66,
  chart_m15_confluence 22.68); fetch gate PASS; skip grep 23; chart counts 81/236, 153/138, 301/309.
- `/compare` vs S_R3: **11 real cells = §2.2 exactly** (H1 `imbalance_instances` 7; both lenses' `fib_lifecycle` 1 and
  `ic_candidates` 1); figures identical (`fig_hash.py` equal to S_R3, hovers included). All 27 CSVs md5-equal to the
  variant `%TEMP%/v65` and its figure hashes equal.
- Docs: `IMBALANCE_FILL_SEMANTICS.md` (the c3 example, both signatures, "The threshold is ONE constant", "Why two
  strokes", the dated shift "Arming threshold 70 % → 65 %"), `POI_ZONES_SPEC.md` §1 + the activation condition,
  `GLOSSARY.md` "fill check", `ARCHITECTURE.md`, `CROSS_CYCLE_FIB_SPEC.md` (two lines), `poi_zones.py` comments; the
  register §D SESSION ORDER item 4 (IN PROGRESS → this plan).

### Step 2 — LANDED 2026-10-21 (`cbaed31`, save S_S2 `20261021_081421_cbaed31`, save commit `780e628`, trunk `f4e83c6`; the user's chart look: "charts look good let's proceed")

- Code: `is_filled`'s four loops (fast path bull / bear, fallback bull / bear) test stroke 2 BEFORE latching stroke 1
  on a candle, so the arming candle's own close is never tested; `_compute_fill_idx_cache` scans stroke 2 from
  `armed_pos + 1` (bull / bear). Docstrings of both.
- Tests: `test_imbalance.py` — the same-candle test flipped into `test_is_filled_the_arming_candle_cannot_also_fill`
  (4 fixtures: bull, bull closing exactly AT the edge, bear, bear at the edge; candle 3 arms + closes past the edge →
  not filled; candle 4 — a RED candle in the bull case — closes at / past it → filled), the same on the label-based
  fallback (a non-RangeIndex frame), and the cache's `(armed, confirmed) == (3, 4)` + cache == `is_filled` at every
  horizon (both index paths). `test_institutional_pattern.py`: `_filled_gap` now asserts the fill lands exactly AT
  `fill_idx`; the user's-sequence test fills imbalance2 at 16 (armed by 15) instead of 15 — candle 15 is the first
  candle after the gap, so no earlier candle could arm it (the one suite failure the change produced; the test's
  intent — imbalance2 filled before the CTS anchor 20 — kept).
- Suite: **1854 passed** (68.9 s; +16 new parametrized cases − the 1 flipped).
- Mutants (`qa_ic_revisit_inputs/mutants_plan_s_step2.py`): **13 / 13 KILLED** — the six same-candle reverts (fast /
  fallback / cache × bull / bear), a half-applied cache edit (the slice moved, the position offset not), and six
  "at the edge" mutants (each `>=` / `<=` of stroke 2 made strict — the user's "On edge is ok"); the bear edge
  mutants are killed ONLY by the bear-closes-at-the-edge fixture (added for that reason).
- Replay: 53.0 s wall (data_fetch 1.56, multi_tf_dual 11.40, chart_h1 6.22, chart_m15_counter 9.78,
  chart_m15_confluence 22.90); fetch gate PASS; skip grep 23; chart counts **81/234, 153/136, 301/307**.
- `/compare` vs S_S1: **51 real cells = §3.2 exactly** (H1 `imbalance_instances` 47, H1 `poi_zones` 2, confluence
  `poi_zones` 2); figures: shapes 236 → 234, 138 → 136, 309 → 307, traces xy-equal; hovers (`cmp_hover.py`): 0
  templates, customdata of 2 traces per chart — the H1 POI sid 1 cycle 2's (IC 859) confirmed candle 998 → 954 on
  the H1 chart and both M15 overlays (= its CSV change). All 27 CSVs md5-equal to the variant `%TEMP%/vS`, figure
  hashes equal.
- **Landing review** (1 read-only lens, conformance + mutation; **197,109 tokens** by the harness): no BLOCKER / MAJOR —
  the three implementations correct in both directions and on both index paths (data-edge clamps included), no fourth
  implementation or same-candle assumption anywhere in production code (landed H1 export: 162 armed, 151 confirmed, 0
  equal; min confirmed − armed = 1), the docs consistent. Folded: **(1, MINOR)** the cache-agreement test ran only the
  label-based fallback (its "range_index" case was an int64 frame) → three index cases now (`range_index` = the fast
  path, `int64_index`, `labelled_index`); **(2, MINOR)** stroke 1 staying latched across a candle that closes back
  INSIDE the gap was unpinned (every fixture filled at armed + 1; a "re-evaluate armed per candle" mutant survived on
  the fallback) → two "armed through a close inside" fixtures (bull / bear, filled at 5) on all three paths + six latch
  mutants; NITs: the cache docstring's "never equal" (→ "when both are set, confirmed > armed"), the activation
  docstring's stroke 2, the fallback's "Verbatim" comment, IMBALANCE_FILL_SEMANTICS' "leave >= end_idx + 2" (→ + 3),
  the stale `1.03` test comments (step 1 leftovers), `_filled_gap` / N17's hard-coded 0.70 (→ the constant). NOT
  folded (pre-existing, unreachable: every engine frame is a contiguous RangeIndex): a `continue` → `break` mutant on
  the fallback's missing-label skip survives, and the cache gives `(None, None)` when the label `end_idx + 1` is
  missing while the fallback skips it.
- After the fold: suite **1868 passed** (90.9 s); mutants **19 / 19 KILLED** (the 13 + the 6 latch mutants); a fresh
  replay (84.9 s wall: data_fetch 2.80, multi_tf_dual 17.40, chart_h1 11.39, chart_m15_counter 15.62,
  chart_m15_confluence 35.32 — a slow run, machine variance; fetch gate PASS, skip grep 23) again md5- / figure-hash-
  equal to the variant. Saved from that replay (`/commit-save` reuse mode — no source changed after it): the save's
  27 CSVs md5-equal to the variant's; `cmp_save.py` vs S_S1 = the 51 cells again (all 47 instance cells are S_S1's
  one-candle fills — none left; no `armed_idx` moves).
- Docs: `IMBALANCE_FILL_SEMANTICS.md` (the predicate table `(armed_idx, check_to_idx]`, "The two strokes fire on
  SEPARATE candles" with the user's words, an edge-case row, "Why two strokes", the dated shift "Separate candles"),
  `POI_ZONES_SPEC.md` (§1 stroke 2 + the activation condition), `GLOSSARY.md` "fill check", `ARCHITECTURE.md`; the
  register §D.2 (the fill check's unreachable missing-label disagreement, above); after the save, the `/compare`
  skill's "PLAN_S step 2" entry + chart counts.

### Step 3 — BUILT 2026-10-21 (the user's chart look + go: "looks good go ahead")

- Code (`zones/poi_zones.py`): `closest_prior_imbalance`, `check_prior_imbalance` (`ICCandleCheck` +
  `prior_imbalance_first_c2_idx` / `prior_imbalance_unfilled`; evaluated on the IC candidates, every check with
  `classify_all`); `select_ic`'s Step 1B flow + `_steps_2_and_3` (today's Steps 2–3 over a list); `ICSelection` +
  `prior_imbalance_outcome` / `prior_imbalance_outranked` (required) + the `ic_candidate_count` /
  `prior_imbalance_passer_count` / `pick_check` properties + the asserts (§4.3 item 4, incl. lens B's pick-is-a-
  candidate); `prior_imbalance_narrowed` (the hovers' one test); the POI meta's 6 keys; `ICCandidateRecord` + the 5
  fields; docstrings (module, `select_ic`, `apply_ranking_rules`, `ICSelection`, `ICCandleCheck`). The mirror
  (`_ZONE_META_IDX_KEYS` + 2, `_IC_CANDIDATE_IDX_FIELDS` + 1), the CSV (`IC_CANDIDATE_COLUMNS` + 5 between
  `ic_zone_band` and `ip_outcome`), the hovers (`_zone_render.py`: `_only_candidate_text`, `_poi_narrowed`,
  `_poi_prior_imbalance_hover_text`; the fib hover's wording).
- Existing tests (the 34 + 20 hand-built metas / the `ICSelection` fixture, §4.5): the shared selection fixture
  `test_institutional_pattern._select` is 1B-NEUTRAL by default (`_bos_gap(bos)` + an assert that passers ==
  candidates; `neutral_1b=False` for the 1B tests) — it alone fixed 21; the direct-`attrs` fixtures got the same gap
  (`test_institutional_pattern` 2, `test_plan_l_landing_2` 3 sites, `test_plan_k_landing_b` 1,
  `test_plan_p_ic_condition3_step1a` 3, `test_plan_h_landing_b._m1_frame` — a gap at its BOS anchor 30); the
  hand-built hover metas `**NO_1B_META` (a fallback, nothing narrowed); `test_poi_meta_carries_…`'s key list + 6;
  `test_event_meta_idx_keys` `must_see` + 2, `NON_INDEX_INT_META_KEYS` + the 2 counts; the plan-L column-position
  asserts; its `ICSelection` `ok` fixture now has its own checks (a pick must be a candidate). No BOS-anchor-IC
  fixture needed special handling. `test_export_header_none_row_and_legs` additionally pins the 5 columns' values.
- New `tests/test_plan_s_step1b.py` (72 tests): the gap's bounds (6 cases), the test through IC − 1 (5), a merged
  gap's first c2, the config's threshold, a degenerate gap; the flow on every path × both selection modes (9 fixtures
  — incl. `_PASSER_PATTERN`, found by a search: narrowed but Step 2 over all picks the passer — sub 8 cross 0→2's
  shape); Step 1A; "never adds or removes an IC" (16 fixtures); `classify_all` vs MS's path; MS's inner; the
  `ICSelection` invariants; the CSV, the POI meta, the mirror; the hovers (6 + the band order + the rejects + the
  fib hover); `prior_imbalance_narrowed`.
- Suite **1940 passed** (143.0 s, a loaded machine) = 1868 + the new file's 72; the touched files' counts unchanged.
- Mutants (`qa_ic_revisit_inputs/mutants_plan_s_step3.py`): **35** — the first run 33 / 35 KILLED; the 2 survivors
  were real test gaps, each fixed by a test: `gap_named_by_its_last_c2` (every test gap was one candle — the window's
  2810..2812 is merged → `test_a_merged_gap_is_named_by_its_first_c2`) and `pick_not_a_candidate_unguarded` (lens B's
  new assert was only ever hit together with others → a case where it alone fires, "with_a_non_candidate"); re-run:
  2 / 2 KILLED. Final re-run of all 35 on the final code (docs + docstrings in place): **35 / 35 KILLED**.
- Replay: 103.6 s wall (data_fetch 3.92, multi_tf_dual 24.67, chart_h1 11.76, chart_m15_counter 17.08,
  chart_m15_confluence 42.58 — every stage ~2× the variant's, candle_features too: machine load; 1B's measured cost is
  ~0.1 s); fetch gate PASS; skip grep 23. **vs the variant `%TEMP%/v1b`** (`cmp_landed_vs_variant.py`): 21 CSVs
  md5-equal, the 6 IC / POI CSVs equal once the 5 columns / 6 meta keys are removed; figures traces + shapes equal
  (all three); hovers (`cmp_hover_strings.py`): exactly §4.3.1's lines — 1142 `only preferred IC candidate` + `Prior
  imbalance 1140 unfilled · preferred 1 of 3 IC candidates`; 2821 (both lenses) + `… 2810 … 1 of 2 IC candidates (over
  pattern 3B.iii at 2808)`; sub 8 cross 0→2's 4131 (both lenses) `Prior imbalance 4118 unfilled · preferred 1 of 2 IC
  candidates`; the fib hovers `… IC 1142 / IC 2821 · only preferred IC candidate`.
- **Landing review — 2 read-only lenses, 369,972 tokens by the harness** (lens 1 conformance + docs 181,396; lens 2
  correctness + tests 188,576). No BLOCKER / MAJOR; the code = the decided rule, flow, names and hovers; every window
  number in the docs re-checked; lens 2 re-verified landed == variant (27 / 27) and audited the landed CSVs for every
  1B invariant (0 violations). Folded — lens 1: the stale doc lines its grep found (POI_ZONES_SPEC "Data Flow",
  Overview, Files, Implementation Status, §3.5's `ip_outcome`; GLOSSARY "institutional pattern" + "Step 1A"; the
  outranked definition "only when the IC came from the passers" in §4 + the `ICSelection` docstring — POI 923, a
  fallback with a pattern, has none; MARKET_STRUCTURE_SPEC: Step 1B's read takes no moment, not `evaluated_at=None`;
  three docstrings; "no prior gap → fails" cited to round 5's final spec, not round 3; §4.3 item 4 as built — the
  `bos_idx <=` bound lives in `closest_prior_imbalance`). Lens 2: 7 new mutants, 6 SURVIVED — N1 / N2 (my edit of
  `test_the_selection_rejects_inconsistent_fields` let the new 1B asserts pre-empt two older ones → every bad case now
  states consistent 1B fields and `match`es its own assert's message), N3–N5 (the three new asserts never fired alone →
  lone cases), N6 (no FILLED prior gap end to end → `test_a_filled_prior_gap_fails_end_to_end`: `two_immediate` with
  gap 8 filled at 10 — 13's row reads c2 8, unfilled False); all 7 added to the mutants file and re-run: **7 / 7
  KILLED** (42 mutants in all). Deferred to `/commit-save`: the register §D item 4 and the `/compare` skill's IC-CSV
  column history (lens 1 item 10).
- After the fold: suite **1941 passed** (137.7 s). **`/compare` vs S_S2** on a fresh replay (107.3 s wall: data_fetch
  3.25, multi_tf_dual 32.42, chart_h1 11.18, chart_m15_counter 16.63, chart_m15_confluence 40.32 — the loaded
  machine; fetch gate PASS; skip grep **24** = the 23 + one benign `[data_bridge] RETRY 1/2 M15 chunk 2025-12-14 …
  504` the fetch recovered from): **= §4.4 exactly** — 21 / 27 CSVs md5-equal, the 6 IC / POI CSVs differ; keyed
  (`cmp_trial.py`): H1 0 changes; counter sub 3 cycle 1 2808 IP 3B.iii → 2821 (confirmed 2829 → 2841); confluence the
  same + sub 0 cycle 2 1216 → 1142 (confirmed 1230 → 1224); structure events 329 / 591 / 1388 unchanged, fibs,
  subs, triggers, KL, WVMI, unresolved unchanged; landed == the variant + the step-3 additions (27 / 27); figures: H1
  identical, chart counts 81/234, 153/136, 301/307 unchanged, the moved POIs' traces only; hovers (`cmp_hover_strings`
  vs S_S2): exactly §4.3.1's lines (+ the per-candle hover rows of the two moved POIs' spans).
- **The chart look (2026-10-21).** Shown: the two moved POIs (confluence sub 0 cycle 2 1216 → 1142, active from 1224;
  sub 3 cycle 1, both lenses, 2808 → 2821, active from 2841), the new hover lines, and the author's defaults listed for
  the user to confirm or change — the count names `prior_imbalance_passer_count` / `ic_candidate_count`, the outranked
  names `prior_imbalance_outranked_ic_idx` / `prior_imbalance_outranked_ip_rank`, a merged run of gap candles = one gap,
  a gap straddling the BOS anchor does not count, no prior gap fails 1B, a zero-size gap counts as filled, the
  outranked pattern by Step 2's tiebreak rules and only when 1B narrowed the field. The user, verbatim: **"looks good
  go ahead"** — every default accepted as built; `/commit-save` reuse mode (no source changed after the `/compare`
  replay).
