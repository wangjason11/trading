# PLAN_S — the imbalance fill rule (65 % arming, separate candles) + IC Step 1B, the closest-prior-imbalance preference (the user's IC revisit, 2026-10-20 → 21)

**Status: steps 1–2 LANDED 2026-10-21** — step 1 source `69c099b`, save **S_S1** `20261021_020427_69c099b` (save
`b4f040b`, trunk `138c2a7`) = §2.2 exactly; step 2 source `cbaed31`, save **S_S2** `20261021_081421_cbaed31` (save
`780e628`, trunk `f4e83c6`; = `LATEST_week9-entries-decisions`) = §3.2 exactly (§6). **Step 3 next, in a new
session** (the author's recommendation, 2026-10-21): a fresh code audit, then §4.3 before any build. Baseline at drafting **S_R3**
`artifacts/commits/week9-entries-decisions/20261020_225749_d424099` (source `d424099`, PLAN_R step 3). HEAD at drafting `896db93`; suite 1839 (green
2026-10-20, 140.7 s). Session order item 4 (register `IDEA_PARKING_LOT.md` §D "SESSION ORDER"). THREE steps, in this
order, each with its own prediction (measured before its build), `/compare` and chart look: §2 the 65 % arming
threshold → §3 the separate-candles fill rule → §4 Step 1B. Commits: the user's call at each `/compare` pause (the
author suggests one per step — 65 % is a global parameter the user may keep without 1B).

Inputs (`qa_ic_revisit_inputs/`): the user's write-up verbatim `user_writeup_20261020.md`; every Q&A round and decision
verbatim + the measurements `qa_findings.md` (rounds 1–5); the read-only shadows `shadow_step1b.py`,
`shadow_fill_rule.py`, `shadow_ic_direction.py` (+ logs); the variant patchers `make_variant_fill65.py` (step 1) and
`make_variant_stroke2_later.py` (step 2, applied on top of step 1's).

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
Round 2 (verbatim): "it should be for all cases when we reach 1B runs and we don't produce a final IC (either fail 1B
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
IC is unfilled by construction). A candidate with no closest prior gap fails 1B (author's default, round 3 — changes
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

### 4.3 Before step 3's build (written after steps 1–2 land, against that code)

The detailed engineering reading (the per-candidate fields, the `ICSelection` outcome, the IC-candidates CSV columns —
a save-format boundary, `/compare` keyed by column name — the POI meta key and the hover line; names proposed for the
user), the FULL-cascade prediction from a variant build on top of steps 1–2 (MS's in-flight reads and the overlay's
capped views can move sub structure events and triggers — first-order at 0.70 under the fallback flow, 19 / 398 M15
MS reads differed), the mutants, and a plan cold review (2–3 lenses, ≈0.6M tokens — stated to the user before it runs).

---

## 5. Reviews and cost

- Step 1: no multi-agent review — a constant, measured exactly by its variant; my 9 mutants.
- Step 2: a landing review, 1 combined conformance + mutation lens (≈0.2M tokens) — three implementations of one rule
  and a flipped pinned behaviour.
- Step 3: a plan cold review (≈0.6M) and a landing review (2 lenses, ≈0.5M).
- Total ≈1.3M subagent tokens, ≤ 6 agents — under the ~2M / 10-agent ask threshold; each is announced with its size
  before it runs.

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
