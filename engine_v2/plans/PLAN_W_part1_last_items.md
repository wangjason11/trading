# PLAN_W — the last Part 1 items: 9a the fill checks require a 0..n−1 frame, 9b the POI record gate for a fib that ended without ever locking, 9c the sub CTS KL zones (SESSION ORDER item 9, the user 2026-10-28)

**Status: 2026-10-28 — 9a LANDED (`35e827e`, = its prediction, zero change, §1.5); 9b LANDED (`aff21f8`, = its
prediction: zero change on the reference window, sub 2 cycle 1's POI on 2025-08-01 → 09-20, §2.5; the user's chart
look: "charts look good, go ahead and commit-save"); one save after both (the user: "Commit now, save after 9b"):
S_W9b `20261028_165600_aff21f8` (save `1ac3e12`, trunk `f798f9e`; = S_V8c byte for byte; = `LATEST_<branch>`). NEXT:
9c (§3, discussion first).** Baseline **S_V8c** `artifacts/commits/week9-entries-decisions/20261028_153518_1318c82`
(source `1318c82`, save `fb6fb81`, trunk `8e6a53a`; HEAD at drafting `6312819`, code = `1318c82`); suite 2172 at the
cold start (89.7 s). The scope is the user's (register `IDEA_PARKING_LOT.md` §D "SESSION ORDER" item 9, their words
verbatim there): 9a → 9b → 9c (proposed order), one cause per `/compare`. Inputs: `plans/plan_w_inputs/`.

---

## 1. 9a — the fill checks require a 0..n−1 frame (zero change)

### 1.1 Today

The imbalance fill rule has three implementations (IMBALANCE_FILL_SEMANTICS "The two strokes fire on SEPARATE
candles"): `ImbalanceInstance.is_filled`'s positional fast path (a 0..n−1 RangeIndex), its label-based fallback loop
(any other index), and `poi_zones._compute_fill_idx_cache` (identity path for an int index with `[0] == 0` and
`[-1] == n − 1`, a label → position dict otherwise). The two label-based paths disagree on a frame with a missing label
(PLAN_S §6 step 2's landing review, register §D.2): the fallback skips it (`continue`; a `continue → break` mutant
survives the suite), the cache answers `(None, None)` when the label `end_idx + 1` is missing.

**Measured (2026-10-28, `plan_w_inputs/w9a_index_census.py` / `.txt`, the CONFIG window with its floor seeds):** all
188,013 `is_filled` calls see a 0..n−1 RangeIndex (the fast path); all 65 cache calls (16,899 instances) take the
identity path — the label-based code is never reached. The disagreement on a synthetic frame (the suite's bullish
fixture labelled 100–105; candle 104 arms, 105 fills — `plan_w_inputs/w9a_disagreement_demo.py` / `.txt`):

| case | `is_filled(t=105)` | cache (armed, filled) | agree? |
|---|---|---|---|
| no label missing | True | (104, 105) | yes |
| label 102 missing (`end_idx + 1`) | True | (None, None) | **NO** |
| label 103 missing (later) | True | (104, 105) | yes |

### 1.2 The decision (the user, 2026-10-28, AskUserQuestion)

Options shown: (1) require a 0..n−1 index in both — raise otherwise, delete the label-based code; (2) the cache follows
`is_filled` (skip a missing label), both paths kept; (3) `is_filled` follows the cache (rejected in the text: a label
gap means nothing); (4) keep, pin only. **The user: "Require 0..n−1"** (option 1, the recommendation — the engine
already assumes label == position in many places, so a non-contiguous frame would break more than these two
functions; one loud check states the assumption once).

### 1.3 The change

- `common/types.py`: a new `require_positional_index(df, caller) -> n` — passes a 0..n−1 RangeIndex (the O(1) duck-typed
  check the fast path has today), an integer index equal to 0..n−1, or an empty frame; raises `ValueError` (naming the
  caller and the index) on anything else. `is_filled` calls it FIRST (before the degenerate-gap return: every call on
  a bad frame raises) and keeps only the positional scan — the fallback loop is deleted.
- `zones/poi_zones._compute_fill_idx_cache` calls it first; the `idx_to_pos` dict, `_pos` / `_idx_from_pos` go (a hit's
  position IS its idx); the `end_idx + 1` past-the-data guard stays.
- **Prediction: ZERO output change** (the census: the deleted code is never reached) — 30 / 30 CSVs + the figures +
  the PNGs byte-equal to S_V8c, the gate + skip grep 33 unchanged.

### 1.4 Tests + mutants (9a)

- `tests/test_imbalance.py`: `test_is_filled_fallback_path_separate_candles` (the fallback on a frame labelled 100..)
  BECOMES the raise test — both checks raise on a frame labelled 100.., on a missing label, and on labels out of order
  with `[0] == 0` and `[-1] == n − 1` (the cache's old identity test passed it); the harm shown: with the check
  bypassed, the positional scan answers "unfilled" for the gap that label 105 fills. The cache-agreement test keeps
  `range_index` + `int64_index` (an int64 0..n−1 frame: accepted, the same answers); its `labelled_index` case moves to
  the raise test.
- Mutants (`plan_w_inputs/mutants_9a.py`): the check removed from either function; the integer-index branch dropped
  (rejects an int64 0..n−1 frame); the cache's old first / last test restored in place of the check.

### 1.5 Result (2026-10-28) — = the prediction

The replay (77.0 s wall: data_fetch 4.88, multi_tf_dual 13.96, chart_h1 8.50, chart_m15_counter 15.60,
chart_m15_confluence 32.35) vs S_V8c: 30 / 30 CSVs md5-equal, the three figures' full JSON equal (79 / 302 / 154
traces, layouts equal — `cmp_fig_full.py`), the three PNGs md5-equal, chart counts 79 / 232, 154 / 134, 302 / 305;
fetch gate PASS (the two `[floor_seed]` lines unchanged), skip grep 33 (`plan_w_inputs/w9a_compare.txt`). Suite 2164
(2172 − the 6 fallback cases − the 6 `labelled_index` cache cases + the 3-case raise test + the accept test); mutants
5 / 5 KILLED (`plan_w_inputs/mutants_9a.py` / `.txt` — the raise test kills both "check removed" mutants and the old
first / last test, the accept test the dropped integer branch and the dropped empty pass). The earlier landings'
anchors in the two files 9a touched (`anchor_scan_9a.py 6312819` → `.txt`): 11 stale of 175, all PLAN_S step 2's
(`qa_ic_revisit_inputs/mutants_plan_s_step2.py`) — 6 on the deleted fallback loop (no code left to mutate) and 5 on
live code 9a moved (the positional scan lost an indent level; the cache's `_idx_from_pos(confirmed_pos)` became
`confirmed_pos`): re-anchored on the new code by `mutants_9a_restated.py`, 5 / 5 KILLED (`mutants_9a_restated.txt`).
The `continue → break` survivor is gone with its loop.
Docs: IMBALANCE_FILL_SEMANTICS "The two strokes fire on SEPARATE candles" (two implementations, the 0..n−1
requirement), GOTCHAS (the positional-scan subtleties — state the 0..n−1 frame instead of a label-based fallback), the
register §D.2 entry → DONE.

---

## 2. 9b — the POI record gate for a fib that ENDED without ever locking

### 2.1 Today (the code — `zones/poi_zones._poi_record_gate` + `select_ic_records`; FIB_LIFECYCLE_SPEC §11)

A fib record gets a POI record iff `(active AND end_idx is None) OR locked` — AND its cycle has a CTS_ESTABLISHED
(option N, the user 2026-09-29). A fib locks at its cycle's CTS_CONFIRMED. So a cycle that ENDS (next cycle, reversal,
a sub's `parent_end` / replacement) before its CTS confirms gets no POI record: no IC rows, no POI; the chart still
draws its fib (faded — the F1 draw gate, `_zone_render.fib_record_drawn`) with the hover `ic_idx=NA (not evaluated: no
POI record)`. A live unlocked fib on the LAST cycle at the data edge passes (`end_idx is None`). Origin: Week 7
`0033c53` (2026-01-31, "active OR locked", `active` then False once a cycle ended); FibState Session 2 `310c395`
(2026-05-27) kept the effect by adding `end_idx is None`. A POI's end is its cycle's end (`compute_cycle_lifecycle`,
inherited in `derive_poi_zones`), so a record that passed the gate would end there with no new code.

### 2.2 Measured (2026-10-28, `plan_w_inputs/w9b_gate_census.py`, `w9b_variant_measure.py`, `w9b_variant_diff.py`,
`w9b_example_table.py`; outputs in `plan_w_inputs/w9b_census/`)

The census (every downstream run, the live version of each cycle that gets no POI record), five NZD_USD windows:
the reference window 2025-12-01 → 2026-01-20 and four others (2025-06-01 → 07-20, 08-01 → 09-20, 10-01 → 11-15,
2026-02-01 → 03-20). In the FINAL records: 0 on the reference window; 0–2 per other window, ALL on subs, never on the
H1 main: Aug–Sep sub 2 cycle 1 (`parent_end`, live 2025-09-08 07:15 → 09-16 17:45), Oct–Nov sub 0 cycle 2
(`parent_end`, 2 candles), Feb–Mar sub 0 cycle 1 (`parent_end`, 3 candles); + two counter-lens cross fibs whose
cycle never had a CTS_ESTABLISHED (Aug–Sep sub 1 cycle 1 ended by reversal, sub 4 cycle 1 open at the data edge —
option N). The PROXIMITY VIEWS (`proximity_overlay.project_sub_as_of`, `cap_reason="proximity_view"`) drop 9–22 per
window — but the overlay reads a sub's POI only from the view capped at its cycle's CTS_CONFIRMED moment (PLAN_R
step 2, "a POI from its cycle's CTS_CONFIRMED event moment"), when the fib is locked: every dropped view record is
in a view read for its BOS zone only. So no gate choice reaches the proximity triggers.

The relaxed gate `active OR locked` (the CTS_ESTABLISHED check kept), base vs variant on four windows: the
reference window ALL EQUAL (0 change); everywhere the structure events, KL zones and subs EQUAL; only new records —
Aug–Sep sub 2 cycle 1 (dual-lens): a POI, IC 2447 (2025-09-05 11:45, a pinbar, pattern 3C.i, band 61.8–80), buy
[0.58720, 0.58740], active 2542 (09-08 11:30) → its cycle's end 3143 (09-16 17:45, `parent_end`), + 362 IC rows (3
IC candidates); Oct–Nov sub 0 cycle 2: a POI, IC 2060 (pattern 3A.iii), active 2337 → 2339; Feb–Mar sub 0 cycle 1:
an IC record with no IC (`no_ic_candidate`, 444 rows), no POI. The example's events (`w9b_example_aug_sub2_c1.txt`):
CTS_ESTABLISHED + BOS_CONFIRMED at 2525 (09-08 07:15), then 45 CTS_UPDATEDs as price kept extending (to 3142,
09-16 17:30), never a CTS_CONFIRMED; the parent cycle ended at 3143. The IC is picked on the fib as of its END (CTS
anchor 3142) while the POI is active from 2542 — as for every POI between its CTS_ESTABLISHED and its lock today,
and for the live last cycle at the data edge.

### 2.3 The decisions (the user, 2026-10-28, AskUserQuestion after §2.2)

Options shown for the rule: (1) a full POI record — IC evaluated + exported, POI drawn, ending at its cycle's end;
(2) IC rows only, no POI; (3) keep as is. **The user: "Full POI record"** (the recommendation — the same treatment a
live unlocked fib at the data edge gets today). Separately, **the CTS_ESTABLISHED requirement: "Keep it"** (a cycle
without one — only a pre-created cross fib — still gets no record: its POI would have no activation floor). And
for the saves: **"Commit now, save after 9b"** — 9a's commit now, one `/commit-save` after 9b lands.

### 2.4 The change + prediction

- `zones/poi_zones._poi_record_gate`: `(active AND end_idx is None) OR locked` → **`active OR locked`** — the cycle's
  condition-active version passes whether or not its cycle ended; still per-record (a superseded cross version or a
  condition-inactive record carries `active=False` and stays out; a locked record always passes). It now equals
  `select_ic`'s own validity check (`not (active or locked)` → no candidates — the Week 7 rule). The CTS_ESTABLISHED
  check in `select_ic_records` is unchanged (the user: "Keep it"); its docstring: reached by a pre-created cross fib
  whose cycle never established, live OR ended. No other reader holds the gate: MS's in-flight read
  (`poi_inner_for_fib`) never had it; the proximity overlay reads a sub's POI only as of its cycle's CTS_CONFIRMED
  (§2.2). The POI's end = its cycle's end (`compute_cycle_lifecycle`, unchanged).
- **Prediction, the reference window: ZERO output change** — measured by the in-process variant (`w9b_variant_diff_
  2025-12-01.txt`: POIs, IC records, events, KL zones, subs ALL EQUAL); so 30 / 30 CSVs + the figures + the PNGs
  byte-equal to S_V8c, the gate + skip grep 33 unchanged. The fib hover's `ic_idx=NA (not evaluated: no POI record)`
  stays only for a cycle without CTS_ESTABLISHED.
- **The chart look — a variant replay on 2025-08-01 → 09-20** (base = `35e827e`, variant = 9b; scratch trees in a
  short `%TEMP%` dir): = §2.2 — sub 2 cycle 1's POI (IC 2447, active 2542 → 3143) on both M15 lenses + its IC rows
  in both `*_ic_candidates.csv`; its fib hover loses the "not evaluated" line; nothing else.
- Tests: `test_institutional_pattern.py::test_records_gate_none_rows_and_tiebreak_on_the_pick` pins TODAY's skip
  (`ended … → gated out`) → its `ended` record now gets its rows (= cycle 0's); + a condition-inactive ended record
  (no rows) so the gate cannot become "every record"; + a new test: the ended record's POI ends at its cycle's end
  (`lifecycle_cap`, `parent_end`) — the harm before 9b: no POI. Mutants: the old gate back; `active AND locked`;
  "every record" (`True`).
- Docs: FIB_LIFECYCLE_SPEC §11 (the POI row + its reconciliation note), CHARTING_SPEC §8 (the "not evaluated" line's
  cause), POI_ZONES_SPEC (where the gate is stated), the register item 9b + D.2 note.

### 2.5 Result (2026-10-28) — = the prediction

- **The reference window** (the repo replay, 65.7 s wall: data_fetch 3.91, multi_tf_dual 13.00, chart_h1 7.45,
  chart_m15_counter 11.26, chart_m15_confluence 28.74) vs S_V8c: 30 / 30 CSVs md5-equal, the three figures' full JSON
  equal, the PNGs equal; fetch gate PASS, skip grep 33 (`plan_w_inputs/w9b_compare.txt`).
- **2025-08-01 → 09-20** (`w9b_window_replay.py`: base = `35e827e`, variant = the working tree; 34.0 / 36.7 s wall; the
  same `[floor_seed]` / fetch lines — H1 1168 / 506, 4 steps; M15 608 / 261, 2 steps; 3440 M15 candles):
  `w9b_census/w9b_aug_sep_replay_compare.txt` — exactly 4 of 30 CSVs differ, rows ADDED only: both lenses'
  `*_ic_candidates.csv` +362 (sub 2 cycle 1, fib 2362 → 3142) and `*_poi_zones.csv` +1 (IC 2447, buy 0.5872–0.5874,
  3C.i, confirmed 2542, `end_idx` 3143 `parent_end`, status `ended`); the H1 chart identical (121 / 153); each M15 chart
  +2 'M15 POI zone' traces + 3 shapes (confluence 204 → 206 / 141 → 144, counter 165 → 167 / 121 → 124) and the
  'M15 fib' trace's customdata — on all 781 points one change only, the sub 2 cycle 1 fib's
  `<br>ic_idx=NA (not evaluated: no POI record)` gone (`w9b_aug_sep_hover_check.txt`); the new POI's hover:
  `ic_reason=IP 3C.i · legs 1-3-1 (2444 | 2445–2447 | 2448)`, `ic_prior_imbalance=2430 unfilled · preferred 2 of 3 IC
  candidates (over pattern 2B.iv at 2501)`, `lifecycle=[2542, 3143] | end_reason=parent_end`.
- Suite 2165 (+ the new POI-end test; the records test's `ended` record now has its rows); mutants 4 / 4 KILLED
  (`plan_w_inputs/mutants_9b.py` / `.txt`).
- Docs: FIB_LIFECYCLE_SPEC §11 (the POI row + the 9b update to its reconciliation note), LANDMINES "FibState
  Lifecycle Gate Is Per-Record" (the POI gate line), POI_ZONES_SPEC (option N's reach), CHARTING_SPEC §8 (the "not
  evaluated" line's one cause), the register item 9 progress.
- **The user's chart look (2026-10-28, the Aug–Sep 9b charts): "charts look good, go ahead and commit-save".**
