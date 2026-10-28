# PLAN_W — the last Part 1 items: 9a the fill checks require a 0..n−1 frame, 9b the POI record gate for a fib that ended without ever locking, 9c the sub CTS KL zones (SESSION ORDER item 9, the user 2026-10-28)

**Status: 2026-10-28 — 9a LANDED (`35e827e`, = its prediction, zero change, §1.5); 9b LANDED (`aff21f8`, = its
prediction: zero change on the reference window, sub 2 cycle 1's POI on 2025-08-01 → 09-20, §2.5; the user's chart
look: "charts look good, go ahead and commit-save"); one save after both (the user: "Commit now, save after 9b"):
S_W9b `20261028_165600_aff21f8` (save `1ac3e12`, trunk `f798f9e`; = S_V8c byte for byte; = `LATEST_<branch>`). 9c
MEASURED + DECIDED (§3.2–3.3: "Export + chart, toggle off"; "Derived, as of the read"); 9c-2 BUILT = its prediction
(zero change, §3.5; committed `321667a` — the user: "Commit now, save after 9c-1"); 9c-1 BUILT = its prediction
(§3.7; the user's look: "thank you this look good. please proceed"). **Item 9 DONE** (+ §4, the KL hover's
`kind=` line, S_W9d). Week 9 Part 1 stays OPEN: the user added item 10, the parked-items sweep (register §D).
One save after 9c-2 + 9c-1: S_W9c `20261028_180228_538cd2f` (source `538cd2f`, save `bfe9f93`, trunk
`58e5cb5`; = `LATEST_<branch>`; the save = `w9c1_landing_check.py` ALL AS PREDICTED vs S_W9b).** Baseline **S_V8c** `artifacts/commits/week9-entries-decisions/20261028_153518_1318c82`
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

---

## 3. 9c — the sub CTS KL zones (discussion first; measured 2026-10-28)

### 3.1 Today (the code)

Every sub projection (`pooled_structure_build.project_to_window` → `orchestrator._run_downstream_pipeline`) DERIVES the
full KL set (`all_kl_zones`, BOS + CTS) — the CTS wave candles are built from the CTS zones — then RETURNS only
`source_kinds=("BOS",)` (since the multi-TF foundation `6a53948`, 2026-02-17). WVMI reads only the BOS zone
(`wvmi._find_bos_zone`), so the sub CTS zones reach nothing else: not the lens `attrs["kl_zones"]`, the KL CSVs, the
M15 charts, nor a probe. The sub probe's reference zone (`entity_df_mutation._build_sibling_cts_ref_zone_from_pool` —
first_counter / subsequent_* — and `_resolve_reversal_start`) is built AD HOC from the winning CTS event
(`reference_zone._derive_cts_zone_ad_hoc`: the same `identify_base_pattern` + `zone_thresholds` as `kl_zones_v1`)
with `kl_zones=[]`; the derived-zone branch (CTS_CONFIRMED winners only) is dead for subs. The H1 main's reversal
reference (`structure_engine`) DOES pass its derived CTS zones (from the reversed sid's events — which end at its
reversal, so its zone as of the read).

### 3.2 Measured (the reference window; `plan_w_inputs/w9c_measure.py` base / cts_export / derived_ref,
`w9c_diff.py` → `w9c_report.txt`, `w9c_derived_ref_diff.py` → `.txt`, `w9c_variant_replay.py`)

**Question 1 — return / export / chart them** (`source_kinds=("BOS", "CTS")`): counter +7 CTS zones, confluence +20
(one per sub cycle with a CTS zone; 4 of them `status inactive` with end = confirmed, e.g. sub 0 c0, sub 1 c0 / c1,
sub 3 c0); events, POIs, IC rows, subs, WVMI, wave candles EQUAL — the variant only adds. The full replay (scratch
tree, 75.4 s): 2 of 30 CSVs differ — the lens `*_kl_zones.csv`, rows added only (confluence 22 → 42 lines, counter
9 → 16); the H1 chart equal; counter 154 / 134 → 192 / 154 traces / shapes, confluence 302 / 305 → 424 / 372; the
gate lines + skip grep 33 unchanged. The look (`w9c_zoom_confluence_980_1800_{today,cts}.png`): red CTS zones over
each sub's breakouts — sub 0 c2's zone (17 expansion steps) a stepped block over Dec 5 → 10.

**Question 2 — the derived zone as the probe reference:** 11 distinct sub reads (6 sibling, 5 reversal), ALL with a
CTS_CONFIRMED winner. The ad-hoc zone = the derived zone's INIT step in 11 / 11 (the same derivation). As of the read
(the sibling window's `hi` / the reversal candle — causal): = ad hoc in 10 / 11; the one difference, the counter read
for the subsequent_confluence probe at 3819 (window 3743–3819, winner sub 6 cycle 0, anchor 3760, confirmed 3786): its
zone expanded once before the read — outer 0.57806 → 0.57827. At the sub's natural end (look-ahead): 2 / 11 differ
(that one → 0.57898 after 9 steps, and the counter read at 3283: 0.57400 → 0.57376). The variant that FEEDS the as-of
derived zone (`derived_ref`): ZERO output change — the 3819 probe ran (cache miss) with the wider zone and returned the
same start (3760, finalized, `no_retrace`, 1 iteration). The H1 main's one reversal read: its derived zone = the ad-hoc
one (0.58534, 0.58404).

### 3.3 The decisions (the user, 2026-10-28, AskUserQuestion after §3.2)

Options shown — question 1: (a) CSV export only; (b) export + chart them; (c) export + chart them, hidden by default
behind a toggle (a small design step first); (d) keep as is. Question 2: (a) keep ad hoc; (b) the derived zone as of
the read; (c) the natural-end zone (rejected in the text: look-ahead). **The user: question 1 "Export + chart, toggle
off"; question 2 "Derived, as of the read".** Two landings, one cause each: 9c-2 (the probe reference — predicted zero
change on the window) first, then 9c-1 (the export + the toggle — CSV rows + figure, the user's chart look).

### 3.4 9c-2 — the sub probe's reference reads the derived CTS zone as of the read (change + prediction)

- `structure/reference_zone.pick_reference_cts_event` — the winner pick lifted out of
  `build_reference_zone_from_cts_event` (which now calls it; no behaviour change): a caller that must know the
  winner before the build reads it there, never a copy.
- `multitf/entity_df_mutation._cts_kl_zones_as_of(sub, cap_abs)` — the sub's slice-local geometry events clipped by
  knowable-at to the cap (`clip_events_to_window`, the projection's clip) → `derive_kl_zones_v1` (the projection's
  derivation; floorless, `lifecycle_cap` = the cap) → its CTS zones; the derivation's debug prints silenced (lines the
  post-replay grep flags re-printed `[probe_ref] …`, `proximity_overlay._VIEW_LOG_KEEP`).
- `_build_sibling_cts_ref_zone_from_pool`: each mirrored event remembers its sub; the winner (`pick_reference_cts_event`
  over the same events and window) — a CTS_CONFIRMED → `kl_zones` = ITS sub's CTS zones as of `hi` (several siblings'
  events compete, all sid 0, so only the winner's sub's zones can be passed); else `[]` (ad hoc, as before).
  `_resolve_reversal_start`: the same with the sub's own events, as of the reversal candle. The H1 main's reversal
  read (`structure_engine`) is unchanged — it already passes its derived zones.
- **Prediction: ZERO output change on the reference window** — the measured `derived_ref` variant (§3.2: every lens
  part equal; the 3819 probe ran with the wider zone, same start); the shadow on the landed code: every sibling read's
  returned zone = §3.2's as-of derived zone (6 / 6, `plan_w_inputs/w9c2_shadow.txt`; the 5 reversal reads' as-of zone = ad hoc). 30 / 30 CSVs + the
  figures + the PNGs byte-equal to S_W9b; skip grep 33 (`run.log` itself gains no grep line — the prints silenced).
- Tests: `test_first_trigger_migration.py` — an autouse recorder replaces `_cts_kl_zones_as_of` for the stub subs
  (`df=None`); (10) `test_primitive_called_with_empty_kl_zones_and_shared_frame` (pinned TODAY's `kl_zones=[]`)
  BECOMES `…_with_the_winners_as_of_cts_zones_and_shared_frame` (two sibling subs: the winner's sub, as of `hi`, its
  zones passed through) + (10b) an UPDATED winner reads no derived zone; `test_reversal_resolver.py` —
  `_expected_reference` follows the rule, `test_cts_kl_zones_as_of_is_the_projection_as_of_that_moment` (= the
  projection's CTS zones with bounds steps; none at 103, one at 104 = its CTS_CONFIRMED moment) and
  `test_the_reversal_reference_reads_the_derived_zone_as_of_the_reversal` (a +20-pip expanded outer reaches the probe;
  the ad-hoc zone keeps the INIT one). Mutants: `plan_w_inputs/mutants_9c2.py` (7).
- Docs: PART4 §4.3.x rows + §17.8, GOTCHAS ("`ref=cts_confirmed` … does NOT mean the derived zone was used" — an
  update note + its Plan C paragraph), `reference_zone` / `entity_df_mutation` docstrings, the register D.2 entry.

### 3.5 9c-2 result (2026-10-28) — = the prediction

The replay (69.7 s wall: data_fetch 6.03, multi_tf_dual 15.65, chart_h1 6.76, chart_m15_counter 11.73,
chart_m15_confluence 28.22) vs S_W9b: 30 / 30 CSVs md5-equal, the three figures' full JSON equal, the PNGs equal;
fetch gate PASS, skip grep 33, no `[probe_ref]` line (`plan_w_inputs/w9c2_compare.txt`). The shadow on the landed
code: 6 / 6 sibling reads return §3.2's as-of derived zone, the 3819 read the wider one (0.57827), every lens part
equal (`w9c2_shadow.txt`). Suite 2168 (2165 + (10b) + the two reversal pins; (10) rewritten); mutants 7 / 7 KILLED
(`mutants_9c2.py` / `.txt` — `v9c2_no_cap` SURVIVED the first run: the as-of test compared bounds only and at R the
reversal is the earlier end; the test now compares the whole zone at 110, where the read's cap ends it, and at R);
the earlier landings' anchors in `entity_df_mutation.py` + `reference_zone.py`: 0 stale of 67 (`anchor_scan_9c2.py
158d3d2` → `.txt`).

### 3.6 9c-1 — export the sub CTS KL zones + chart them behind a toggle, hidden on load (change + prediction)

- `multitf/entity_df_mutation.py` (the lens projection, `project_to_window(...)` for a unique sub): `source_kinds=
  ("BOS", "CTS")` — the sub's CTS KL zones reach the lens `attrs["kl_zones"]` (mirrored like the BOS ones) and so the
  lens `*_kl_zones.csv`. The proximity overlay's views keep the default `("BOS",)` (it reads BOS zones only). The other
  readers of a lens's KL zones are unaffected (checked): `collapsed_cycles` and the wave-candle `cycle_life` read BOS
  zones only; `struct_direction_by_sid` reads the direction (the same on both kinds); WVMI reads only the BOS zone.
- `charting/export_m15_chart.py`, the sub KL loop: a CTS zone is drawn exactly like a BOS zone (style, tier, fills
  per active stretch, the stepped outline, the confirm line, the hover lines; a collapsed cycle's skipped) but every
  trace and shape it adds gets `legendgroup = SUB_CTS_GROUP = "sub_cts_zones"` and `visible = False`. The hover = the
  KL zone hover (`KL Zone`, `side=` — as the H1 chart's CTS zones; no BOS / CTS label on any chart).
- The control bar (`chart_controls.js`, `export_plotly._chart_controls_cfg(..., sub_cts_group=)`): on the M15 charts
  a second button **`Sub CTS zones: hidden`** next to `Parent (1H)`, flipping that group (hidden ↔ shown); the
  toggle code shared by both buttons. AUTHOR'S DEFAULTS (the user's look decides): the label, the place, hidden on load.
- **Prediction** (the measured variant, §3.2, + the hiding): vs the 9c-2 replay (= S_W9b): exactly 2 of 30 CSVs differ —
  the lens `*_kl_zones.csv`, rows added only (confluence +20, counter +7); the H1 chart identical; each M15 chart
  ADDS the variant's traces + shapes (counter +38 / +20, confluence +122 / +67), every one in `sub_cts_zones` and
  hidden, and the control script gains the button — every other trace / shape equal by name; the PNGs byte-equal
  (hidden); skip grep 33. (`debug/chart_census.py` counts the hidden traces too.)
- Tests: the projection returns CTS zones (a lens's KL zones hold both kinds; the overlay's view BOS only); the M15
  chart tags + hides exactly the CTS zones' traces / shapes; the control cfg + script carry the group and the button.
- Docs: CHARTING_SPEC §14 (the button) + §5 (sub KL zones: BOS shown, CTS hidden behind it), ARCHITECTURE ("KL
  zones are BOS-only for subs" → BOS + CTS returned since 9c-1), PART4 §17 (the projection's `source_kinds`), the
  orchestrator comment ("For sub callers `source_kinds=["BOS"]`").

### 3.7 9c-1 result (2026-10-28) — = the prediction

The replay (76.0 s wall: data_fetch 5.14, multi_tf_dual 12.46, chart_h1 6.65, chart_m15_counter 12.48,
chart_m15_confluence 38.03) — `plan_w_inputs/w9c1_landing_check.py S_W9b artifacts/debug artifacts/charts
<the §3.2 variant tree>` → `w9c1_landing_check.txt`, ALL AS PREDICTED: the two lens `*_kl_zones.csv` md5-equal to the
variant's (rows added only: confluence 22 → 42, counter 9 → 16), the other 28 md5-equal to S_W9b; the H1 figure equal;
each M15 chart's elements outside `sub_cts_zones` = S_W9b's, the group all hidden and = what the variant added
(confluence +122 traces / +67 shapes → 424 / 372, counter +38 / +20 → 192 / 154), the HTML names the group; the PNGs
equal. Fetch gate PASS, skip grep 33. In a headless browser (`w9c1_headless_check.py <confluence chart> <shots> 1400` →
`w9c1_headless_check.txt`, `w9c1_headless_{1_hidden,2_shown}.png`): on load `Sub CTS zones: hidden [off]` with
122 / 122 traces + 67 / 67 shapes hidden; a click shows them all (`shown`), a second hides them again; `Parent (1H)`
still flips its own 56 + 38 only. Suite 2172 (+ the projection pin + 3 in `test_plan_w_sub_cts_zones.py`; the
control cfg test's dict gains `subCtsGroup`); mutants 6 / 6 KILLED (`mutants_9c1.py` / `.txt` — `v9c1_shapes_
untagged` first "killed" by a SyntaxError — an empty loop body — re-run with `pass`: killed by the hidden-elements test). The earlier landings' anchors in the files 9c-1
touched (`anchor_scan_9c1.py 321667a` → `.txt`): 8 stale of 285, all the control bar's own (`qa_chart_edits_inputs/
mutants_chart_controls.py`) — 4 slice mutants of `_tag_parent_overlay` (its loop lines now also in `_tag_sub_cts_zone`)
and 4 on the M15 control-config call (it gained `sub_cts_group=`): re-anchored by `mutants_9c1_restated.py`, 8 / 8
KILLED (`.txt`).
- **The user's chart look (2026-10-28, the reference window's M15 charts, the button): "thank you this look good. please
  proceed"** — the label `Sub CTS zones`, its place next to `Parent (1H)` and hidden-on-load as built.

---

## 4. The KL zone hover's `kind=` line (the user 2026-10-28, after item 9)

Raised at the session's close (9c-1 left the sub CTS zones on the KL zone hover, which never said BOS or CTS on any
chart — `side=` vs the structure's direction told them apart). Offered as an idea for the backlog; **the user: "yes
we should add that to KL zone hovers. For consistency it should be done for both main and subs"**. The label (options
shown with a rendered hover each: `source_kind=` — the CSV column / code field, recommended; `zone_kind=`; `kind=`):
**the user: `kind=BOS`**, right after the `KL Zone` title.
The label matches the BOS / CTS dot hover's own `kind=` line (CHARTING_SPEC "Dot hover": `kind=BOS`).

**The change:** `charting/_zone_render.kl_zone_hovertemplate` gains `kind=%{customdata[10]}<br>` right after `KL Zone`;
`kl_zone_customdata` appends `str(zone.source_kind)` as slot 10 (slots 0–9 keep their places for any reader) and fails
loudly on a kind other than BOS / CTS. One template + one row builder serve every KL zone hover: the H1 chart, the M15
charts' H1 overlay and every sub zone (incl. the hidden CTS ones). **Prediction: FIGURE-ONLY, the KL hover traces
only** — counted on S_W9c (`w9d_kl_hover_count.py` → `w9d_kl_hover_count_S_W9c.txt`): H1 26, confluence 208 (106
hidden CTS), counter 72 (32 hidden CTS) change in `hovertemplate` + `customdata` only; 30 / 30 CSVs, every other trace,
the layouts (shapes) and the PNGs equal.

**Result (2026-10-28) — = the prediction.** The replay (78.3 s wall: data_fetch 6.03, multi_tf_dual 12.66, chart_h1
6.39, chart_m15_counter 12.96, chart_m15_confluence 38.96), `w9d_landing_check.py S_W9c` → `w9d_landing_check.txt`
ALL AS PREDICTED: CSVs 30 / 30 md5-equal, PNGs equal, layouts equal; changed exactly the KL hover traces — H1 26 (BOS
6, CTS 20), confluence 208 (sub BOS 76 shown, sub CTS 106 hidden, overlay BOS 6 / CTS 20), counter 72 (14 / 32, 6 /
20), each template = S_W9c's + the one line, each customdata row = S_W9c's + the kind; a sub hover is `kind=CTS`
exactly when hidden. `debug/chart_census.py` S_W9c vs current: 0 deltas, no unknown key, both lenses (the GOTCHAS
rule for a hover-format change). Gate PASS, skip grep 33. Suite 2173 (the KL hover pins + one new); mutants 4 / 4
KILLED (`mutants_9d.py` / `.txt`). A hover-format boundary (figure-only; positional `cmp_save` fine, a hover diff shows
every KL hover changed).
The user (after the look, AskUserQuestion): **"Commit + /commit-save now"** — its own save: **S_W9d
`20261028_182318_7c9477e`** (source `7c9477e`, save `c02c567`, trunk `050bf18`; = `LATEST_<branch>`; the save =
`w9d_landing_check.py` ALL AS PREDICTED vs S_W9c). (`w9d_kl_hover_examples.py` / `.txt` — today's H1 KL hover for one
BOS and one CTS zone, the kind readable only from `side=` vs `struct_direction=` — appeared in this folder during
the label question; its origin was not traced: kept, it is accurate and on topic.)
