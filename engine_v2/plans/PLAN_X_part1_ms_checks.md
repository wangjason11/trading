# PLAN_X — the item-10 picks: the REVERSAL_CANDIDATE pattern fix + the MS risk census (Week 9 Part 1)

**Status: X1 LANDED `6764bde` = its prediction (§X1 "Result"), SAVED S_X1 `20261028_195331_ebbf223` (source
`ebbf223`, save `b5c61de`, trunk `96def92`; = the landing replay byte for byte) · X2 MEASURED (§X2 "X2 result": every
check 0 in effect on 5 windows) · X3 the fib re-ask built as a VARIANT — asked for by the user 2026-10-28 ("Build the
fib re-ask as a variant"); built + replayed on 5 windows (§X3: the user's picks "All four sites", no cold review) —
0 MS decisions flip (§X3 "X3 result"); the user: "Close the line" (not built). PLAN_X COMPLETE — Week 9 Part 1
CLOSED 2026-10-28; the branch pushed after the close (the user: "Push after the close").** Register `IDEA_PARKING_LOT.md` §D SESSION
ORDER item 10 — the user's pick 2026-10-28 (verbatim there and in `part1_sweep_inputs/part1_sweep.md` "The user's
pick"): "Measure 3 MS risks", "Fix the '?' pattern meta", "Re-check parked latent cases". The sweep list's numbers
(1.1, 1.3, …) are its section numbers. Baseline: S_W9d `20261028_182318_7c9477e` (HEAD `93e8304` = S_W9d's code +
docs). Inputs folder: `plans/plan_x_inputs/`.

Order: **X1** (one small landing, its own `/compare`) → **X2** (measurement only — no engine change; its results go to
the user, each fix a later plan of its own). Then Part 1 closes (the user's call) → the push (the user's call) →
Part 2.

---

## X1 — `REVERSAL_CANDIDATE.meta["pattern"]` = the pattern's name (sweep 1.5)

**Today (code at `93e8304`).** `MarketStructure._schedule_reversal_from_anchor` (`structure/market_structure.py:811`)
names the scheduled reversal pattern with `getattr(ev_r, "pat", None) or getattr(ev_r, "pattern", None) or "?"`, and
`_maybe_apply_pending_reversal` (`:848`) does the same for its `[RV_APPLY]` debug line. `ev_r` is a `PatternEvent`
(`common/types.py:31`), whose name field is `name` — it has neither `pat` nor `pattern`, so both always give `"?"`.
The STATE_CHANGED → reversal event's `meta["pat"]` reads `ev.name` (`:1321`) and is right. Found 2026-09-19 (Plan A's
L4 verifier; GOTCHAS "`REVERSAL_CANDIDATE.meta["pattern"]` Is Always `"?"`").

**The change.** Both sites read `ev_r.name` / `st.pending_reversal_ev.name` — direct access, no fallback (the event
contract: `detect_best_for_anchor` returns a `PatternEvent`; a missing name is a bug, not a `"?"`). The meta key and
its meaning are unchanged (it always meant the pattern's name; the value was wrong) — not a key migration (LANDMINES
"Event Contract Rules" rule 3 covers renames / meaning changes).

**Readers of the value** (grep `REVERSAL_CANDIDATE` + `"pattern"`): the H1 chart's reversal-watch marker hover
(`charting/export_plotly.py` ~:1640, `pattern=%{customdata[3]}` from the candidate keyed by `pattern_anchor_idx`); the
events CSV export. No engine decision reads it. The M15 charts draw no reversal-watch marker. The test fixture
`tests/test_lifecycle_sweep_predicted_table.py:118` hand-copies the H1 candidate with `pattern="?"` (no assertion
reads it) → updated to `"continuous"` with the fix, to stay a faithful copy.

**Prediction vs S_W9d** (from S_W9d's own events: each candidate was realised by the STATE_CHANGED → reversal that
applied the same `PatternEvent` — `plan_x_inputs/x1_predict_pattern.py` → `x1_predict_pattern_output.txt`):
- the 3 structure-events CSVs: exactly 6 cells, each REVERSAL_CANDIDATE's meta `'pattern': '?'` →
  - H1 `structure_events.csv`: 897 (apply 902) → `continuous`;
  - confluence: sub 0 1936 (apply 1940) → `one_maru_continuous`; sub 1 2465 (2470) → `continuous`; sub 2 2824 (2829)
    → `one_maru_opposite`; sub 4 3588 (3589) → `one_maru_continuous`; sub 7 4196 (4200) → `one_maru_opposite`;
  - counter: 0 candidates → byte-equal;
- the other 27 CSVs byte-equal (no reader);
- the H1 figure: only the reversal-watch marker trace's customdata slot 3 at the idx-897 point(s), `?` → `continuous`
  (every other watch point shows `none` — no candidate); the two M15 figures equal; PNGs equal;
- run.log: the 6 `[RV_SCHEDULE] … pat=?` lines (H1 1 + 5 sub geometry runs, slice-local indices) name the pattern;
  `[RV_APPLY]` prints 0 lines today (no change);
- skip grep 33; `[floor_seed]` H1 840 / 355, M15 816 / 310; charts' trace / shape counts unchanged.
Not a save-format boundary (same columns, same keys).

**Tests.** A pin on a fixture with a scheduled reversal: the REVERSAL_CANDIDATE's `meta["pattern"]` equals the
STATE_CHANGED → reversal's `meta["pat"]` and is not `"?"` (the realised pattern = the scheduled one). Mutants: restore
the `getattr` chain at :811 (the pin must fail — the harm: `'?'`); `:848` is log-only (debug print) — pinned by a
capsys check on `[RV_APPLY]` if a fixture applies through `_maybe_apply_pending_reversal`, else recorded as
log-only.

**Docs (same commit).** GOTCHAS entry → "FIXED 2026-10-28" (+ the one-line why); register §D.2 line → "Recently
closed"; PLAN_A §8's bullet gets the closing pointer; the sweep list's 1.5 status.

**Result (2026-10-28) — = the prediction.** Tests first: the 3 pins failed on the old code with `'?' ==
'one_maru_continuous'` (both paths) and the `pat=?` log line, then passed; suite **2176** (2173 + 3). Replay 101.7 s wall;
FETCH GATE PASS; skip grep 33; the 6 `[RV_SCHEDULE]` lines name their pattern (`[RV_APPLY]` still 0 lines). `/compare`
vs S_W9d (`cmp_save.py` + `fig_named_diff.py` + md5): **28 / 30 CSVs + the 3 PNGs byte-equal**; the 2 events CSVs differ
by exactly the 6 predicted cells; the figures: counts equal (79 / 232, 192 / 154, 424 / 372), x / y and shapes equal, the
only change the H1 `reversal candidate` trace's customdata — its ONE point, 897: `?` → `continuous`
(`plan_x_inputs/x1_landing_check.py` → `x1_landing_check.txt`; the prediction's "other watch points" do not exist — the
trace has one point). Mutants (`plan_x_inputs/mutants_x1.py` → `mutants_x1.txt`, full suite each): **4 / 4 KILLED**
by the X1 pins — the old chain at the schedule site (3 failing: both paths' candidate + the log line), the old chain
at the apply site (the `[RV_APPLY]` line), a constant name (3), the meta fed from the direction (2).

---

## X2 — the MS risk census (measurement only; sweep 1.1, 1.3, 2.1 + the latent re-checks 1.2, 1.6, 1.7, 1.8)

One script, `plan_x_inputs/x2_ms_census.py START END OUT.txt`, after `plan_w_inputs/w9b_gate_census.py`'s pattern:
fetch the window's H1 history + floor seeds, call `run_replay.run_pipeline` in-process with the shadows installed
(wrappers that call the original and RECORD — never change a return value), write nothing to `artifacts/`. Each check
prints how many cases it EXAMINED, so a 0 means "0 of N", not "never ran". Windows: the reference window 2025-12-01 →
2026-01-20 and PLAN_W's four others (2025-06-01 → 07-20, 08-01 → 09-20, 10-01 → 11-15, 2026-02-01 → 03-20).

**1.1 — the probe's BOS_0 read past its bound.** Wrap `unified_probe._run_phase1` / the Phase 2 routine to know each
call's `probe_end_idx`, and `unified_probe._bos0_inner_at_start` (every reset re-derives the moving BOS_0 at the new
start). At each call: the (inner, outer) on the frame as passed vs on `df.iloc[:probe_end_idx + 1]` (the frame is
0..n−1, so position = label — PLAN_W 9a). Report per window: resets examined; resets whose anchor + 5 > the bound
(exposed); resets whose zone DIFFERS when cut (the cases that matter), with the probe's trigger / entity / candles.
The first_confluence CONSTANT reference zone (`build_ad_hoc_bos0_reference_zone` at the input candle) is checked the
same way. A differing zone means the probe's start could move under the stated bound — then the fix's prediction is a
variant replay (its own plan).

**1.3 — MS's POI-inner snapshot vs as of the candle.** What can go stale (read from the code): `select_ic` takes no
moment — the IC pick's fill questions are asked as of the fib's CTS anchor (by decision, LANDMINES "Bounded MS
Runs…"), which a refresh already follows; only the embedded tracker's REPRESENTATIVE FIB of the cycle and the IC
REFERENCE (`tracker.ic_reference(..., evaluated_at=)`) can change between two refreshes. Wrap
`MarketStructure._check_proximity_at_candle` (it runs only on candles past the CTS while the phase is
established / updated, the gaps pass and the carve-out allows): recompute the inner as of candle i — the same call
`_refresh_poi_inners_for_cycle` makes, with `evaluated_at = i` and the tracker's current representative fib — and
evaluate the same proximity test with it. Report: candles examined; candles whose as-of inner differs; candles whose
HIT differs (the decision); per (run, sid, cycle) only the FIRST differing decision (the paths diverge after it).

**2.1 — fibs dropped when the gap completes a candle late.** Wrap `has_unfilled_imbalance` as FibTracker
(`fib_tracker._has_unfilled`: the CTS_ESTABLISHED activation :853, the first activation on a later update :1110, the
update's re- / deactivation :1218) and `cross_cycle_fib` (the target cycle's own gap :273, the walk-back :305) call
it. When it answers False, re-ask with `evaluated_at` + k for k = 1..5 (the c3 rule: a gap overlapping the window
forms at most one candle after its last c2): the first k at which it answers True = an exposure ("would succeed k
candles later"). Report: False answers examined; exposures by site and k; and, from the run's final fib records, for
each exposed (tracker, sid, cycle): did the cycle get its fib anyway (activated at a later CTS_UPDATED — a DELAY) or
not (a DROP)?

**Latent re-checks (from the run's final event streams + the pool tables):**
- 1.6 a kept REVERSAL_CANDIDATE on a sub whose `apply_idx` > the sub's `end_idx` while the sub ended for another reason
  (`parent_end`, `same_dir_replacement`);
- 1.7 (B5) a BOS_CONFIRMED whose moment (`confirmed_at`) == its structure's STATE_CHANGED → reversal candle (+ the
  broader: a BOS confirmed after the reversal watch froze its barrier, before the apply — informational);
- 1.8 two RANGE_STARTED with the same `range_anchor_idx` in one structure stream (the fixture case: the offline
  finalize re-opens a range the proximity path opened); the detector is checked on that fixture
  (`tests/…_make_double_rewind_data`) first — it must flag it;
- 1.2 each bounded MS run (the first_confluence probes' Phase 2, `end_idx = probe_end_idx`): the full-frame
  `attrs["imbalances"]` instances formed by the bound vs `compute_imbalance` on the frame cut at the bound — count the
  instances whose degeneracy differs (the only way the residual reaches an output, LANDMINES "Known residual").

**Outputs.** `plan_x_inputs/x2_census_<START>_<END>.txt` per window + `x2_census_summary.txt`; the tables shown to the
user = these files. **Cost:** five in-process replays (~1–2 min each); solo, no subagents.

### X2 result (2026-10-28, HEAD `6764bde`; `x2_census_summary.py` → `x2_census_summary.txt`)

Five windows (H1 840 / 861 / 789 / 1058 / 819 candles; every fetch and floor seed complete — the per-window files
print them). The fixture self-check passed on every run (the range-reset detector flags `_make_double_rewind_data`'s
(9, 13, 12)). Totals:

| Check | Examined | Result |
|---|---|---|
| 1.1 the probe's BOS_0 vs its bound | 26 resets + 9 ad-hoc reference zones | 0 resets with `candidate + 5 > probe_end_idx`; 0 zones differ when the frame is cut at the bound |
| 1.3 MS's POI snapshot vs as-of | 6,326 checked candles, 4,656 with a POI | 0 inners differ, 0 decisions differ |
| 2.1 a gap one candle late | 236 False answers → 14 distinct decisions (downstream; the MS-embedded tracker saw the same ones) | 0 activations DROPPED — see below |
| 1.2 bounded-run gaps | 8 bounded MS runs (the FC probes' Phase 2), 2,540 gaps formed by the bound | 0 degeneracy differs (1 merged-bounds difference, Aug–Sep: harmless by LANDMINES' own analysis) |
| 1.6 the cap-straddling candidate | 12 kept REVERSAL_CANDIDATEs | 0 |
| 1.7 B5 | 13 reversals (32 structure streams) | 0 (and 0 BOS confirmed after a watch froze its barrier) |
| 1.8 the double RANGE_STARTED | 138 RANGE_STARTED | 0 duplicates (superseded since PLAN_J J1 — its fixture no longer emits one); 1 range reset before its confirmation moment (below) |

**2.1 — the 14 decisions.** 5 ACTIVATIONS, each one candle late (the fib activated at the next candle's CTS update:
53 → 54, 61 → 62, 60 → 61, 54 → 55, 55 → 56); 5 UPDATE re-checks — 3 a de-activation one candle before the
re-activation (133 / 134, 56 / 57, 55 / 56: successive CTS updates) and 2 on the H1 main sid 0 cycle 0 (deactivated
at 117, reactivated at 172 though a gap formed at 164 — the cycle-0 carve-out keeps MS's proximity off there); 4
CROSS decisions — 2 one candle late (cycles 4 and 1: activated at moment + 1) and 2 crosses PRE-CREATED for a cycle
not yet established: reference window counter sub 6 at M15 3806 (cycle 1 never establishes; the cross only forms at
4179, after the sub's end 4143 — exactly Plan D §7.3's case) and Oct–Nov sub 1 at M15 2210 (cycle 3 establishes at
2259, where the cross re-activated; confirmed by pullback at 2264; all before the sub's live start 2339 —
`x2_cross_case_check.py` → `x2_cross_case_2025-10-01_sub1.txt`). A pre-established cycle has no POI (option N,
2026-09-29) and no CTS for MS to confirm. Not proven here: whether a one-candle exposure at a CTS update ever flips a
proximity decision (the 1.3 shadow cannot see it — it reads the tracker's fib as decided) — answerable exactly only by
a variant with the re-ask built.

**1.8 — the one range reset before its confirmation** (reference window confluence sub 7, cycle 0, sell;
`x2_range_reset_case.py` on S_W9d): the 3813 breakout; the range label's range from 3814 is confirmable at 3819 but
in force from 3814 (RANGE_UPDATED 3815); the 3817 breakout resets it (`range → breakout`). This is the documented
range-label look-ahead (MARKET_STRUCTURE_SPEC "the range confirmed at `c` is active from `i`"; J1 undid it only for
the RE-STEPPED candles) — a live-mode question, with §F's live-mode pattern selection.

---

## X3 — the fib re-ask, built as a VARIANT (PLAN_F §7; the user 2026-10-28: "Build the fib re-ask as a variant")

A VARIANT: patched scratch trees only (`plan_x_inputs/x3_variant_patch.py`), the main checkout untouched; base = HEAD
`8852f00` (engine code = S_X1's `ebbf223`). The question it answers: does a fib decision one candle late ever flip
an MS decision? A real landing would be its own plan / prediction / `/compare`.

**Design input (measured, `plan_x_inputs/x3_design_tables.py` → `x3_design_tables_<window>.txt`, read-only, X2's
harness).** For each of X2's 14 exposed decisions: is the SAME tracker fed another event for the cycle at m + 1 (the
gap's c3)? **11 / 14 yes** — that event asks the question itself (all 5 late activations, all 3 de- / re-activations,
the Oct–Nov cycle-4 and Feb–Mar cycle-1 crosses, the H1 171 update); a re-ask at m + 1 = today. **3 / 14 no** — none
read by MS: (1) reference H1 main sid 0 cycle 0, the update at 163 (gap c1 162 / c2 163 / c3 164, 0.56147–0.56212;
next fed event the update at 170) — the cycle-0 carve-out keeps MS's proximity off; (2) reference counter sub 6, cycle
1's cross BEFORE its establishment, the threshold check at local 235 = M15 3806 (gap c3 3807, 0.57596–0.57604; next
threshold event 608 = M15 4179, after the sub's end 4143); (3) Oct–Nov sub 1, cycle 3's cross before establishment,
the threshold check at M15 2210 (c3 2211; next event the establishment at 2259; before the sub's live start 2339).
(The False-answer counts are X2's minus 1 per window: X2 ran its fixture self-check first.)

**The user's picks (2026-10-28):** scope "All four sites" (incl. the pre-established cross on threshold updates — the
counter sub 6 case); "No, build the variant" (no cold review — a landing would get its own).

**The rule (as code).**
1. *Which decisions:* every FibTracker imbalance decision answering "no unfilled sd gap" over a window that ends AT
   its moment m (the only shape a later-forming gap can enter: a gap with c2 = m forms at c3 = m + 1) —
   (a) the cycle's single at CTS_ESTABLISHED (cycle 0 / past the cross-fib limit; `_handle_cross_cycle_cts_established`
   no-cross branch, the `has_unfilled` of `_on_cts_established` ~:853); (b) that single's first activation at a
   CTS_UPDATED (~:1110); (c) an existing single's re- / de-activation at a CTS_UPDATED (`_update_fib_cts` ~:1218);
   (d) the cross check's own-gap test (`_cross_check` → `resolve_cross_cycle_eligibility` ~:273; from
   CTS_ESTABLISHED / CTS_UPDATED and, before establishment, CTS_THRESHOLD_UPDATED). The record is registered only when
   the window reaches the moment (`hi >= m`), keyed `(sid, cycle)` (one pending per key; a cross registration
   suppresses the inner single update's). NOT re-asked: the cross walk-back over earlier cycles (~:305 — its windows end
   at an earlier CTS anchor; X2: 0 exposed) — a counter logs any walk window reaching the moment.
2. *When:* once, at `due = m + 1`, as of `due` — the knowability cut and the fill horizon both `due` (every site's
   rule since Plan E E3a), the same window and anchors. False again → done (nothing else forms in a window ending at
   m).
3. *Order:* `flush_reasks(upto, df)` runs every pending re-ask with `due <= upto` (in `(due, registration)` order)
   BEFORE a fed event whose moment `>= due` is handled — so an event at m + 1 decides on top of it (= today in the
   11). A fed event for the same key at a moment `< due` (the same candle m: a lock, a newer question) cancels it;
   CTS_CONFIRMED of the key always cancels (nothing activates after the lock). Dropped when `due` is past the
   structure's end (the downstream feed: the sid's reversal candle; MS: its run stops there) or the data edge / a
   bounded run's end (never flushed).
4. *Re-runs:* (a)/(b) activate the single if still absent and the window now holds an unfilled gap
   (`activated_at = due`, `activated_on = "reask"`); (c) reactivate if still inactive, unlocked, same anchor
   (`reactivated_at = due`, `reactivated_on = "reask"`); (d) re-run `_cross_check` with the stored arguments at `due`
   unless the target is confirmed (records created / revived carry `activated_on` / `reactivated_on = "reask"`).
5. *Feeders:* the downstream feed (`orchestrator`, the fib loop) flushes before each fed event and once at the end
   (`upto = len(df) - 1`), with the sid ends from `compute_reversal_idx_by_sid`; MS flushes in `_feed_fib_tracker`
   (before each fed event, at its moment) and at the start of every candle step (`_replay_step_no_patterns`) — then,
   if a re-ask changed the CURRENT cycle's fib before confirmation (`cts_phase == "EST_OR_UPD"`), refreshes its POI
   snapshot at that candle, before the proximity check (as a CTS_UPDATED at that candle would). Same code in both
   trackers (`check_processing_order` already pins their feed orders equal).

**Prediction (before code), vs the base trees (HEAD) on the 5 windows:** **0 MS decisions flip** — no structure
event (CTS / BOS / reversal / range), trigger, sub or IC pick moves (the 11: the same state before m + 1's proximity
check; the 3: unread by MS). Downstream only: counter sub 6 (reference) — its cycle-0 fib ends at M15 3807
(`new_cycle`, the existing "a pre-established cross ends the previous cycle's fib at its creation") instead of 4143,
and a cycle-1 cross record from 3807 (a drawn fib if the chart gate keeps it); H1 main sid 0 cycle 0's fib meta
(`deactivated_at` 117 → 170; active 164–170); Oct–Nov sub 1's cross meta (pre-live); the 11 — fib meta labels only
(`activated_on` / `reactivated_on = "reask"`). Anything else = STOP and name the mechanism.

**Procedure.** `x3_variant_patch.py TREE` (exact replacements, each asserted once) + `x3_variant_replay.py` (base and
variant trees per window, `%TEMP%` short paths, the trees deleted after — each holds `oanda.cfg`); diff keyed
(`x3_variant_diff.py`: the structure events, triggers, subs, IC / POI / fib records) → `x3_variant_diff_<window>.txt`
+ `x3_summary.txt`; timing blocks shown.

### X3 result (2026-10-28, base HEAD `8852f00`) — = the prediction: 0 MS decisions flip

10 scratch-tree replays (5 base, 5 variant; all exit 0; wall 59–187 s run 10-parallel — reference 186.7 s base /
186.0 s variant). `x3_summary.py` → `x3_summary.txt` (per window `x3_variant_diff_<window>.txt`):

| Window | CSVs md5-different | MS-decision rows (structure events / triggers / subs / unresolved) | fib records differing | charts |
|---|---|---|---|---|
| 2025-06-01 → 07-20 | 1 / 30 | 0 | 1 (meta label) | equal |
| 2025-08-01 → 09-20 | 2 / 30 | 0 | 2 (meta labels) | equal |
| 2025-10-01 → 11-15 | 1 / 23 | 0 | 2 (meta; sub 1 cycle 3's `reactivated_at` 2259 → 2211, pre-live) | equal |
| 2025-12-01 → 2026-01-20 | 3 / 30 | 0 | 6 (below) | counter: +1 `M15 fib` trace, shapes 154 → 159 |
| 2026-02-01 → 03-20 | 1 / 23 | 0 | 3 (meta labels) | equal |

Only `*fib_lifecycle.csv` files differ; every structure-events, trigger, sub, POI, KL, IC, WVMI and imbalance CSV is
byte-equal on all 5 windows. The **14 distinct re-asks that changed a fib are exactly X2's 14 decisions.** On the
reference window: counter sub 6's cycle-0 fib ends at **3807 `new_cycle`** (was 4143 `same_dir_replacement`) and a
**cycle-1 cross record 3807 → 4143** appears (`active`, unlocked, `activated_on = "reask"`) — the one drawn change;
H1 main sid 0 cycle 0 — `deactivated_at` 117 → 170 (active 164–170; the cycle-0 carve-out); 3 meta labels. The
cross walk-back: 0 windows reaching the moment (of 93–939 questions per window, process totals) — not re-asked,
confirmed unexposed.

**The MS side ran** (`x3_ms_reask_check.py` in each variant tree → `x3_ms_reask_check_<window>.txt`; counting wrappers
only): the embedded trackers' re-asks CHANGED a fib 1 / 4 / 2 / 6 / 6 times per window (all MS runs, probes and
look-ahead re-steps included); 22 MS POI refreshes came from a re-ask, 9 with a different inner (`[]` → an inner) —
every one at a candle whose own CTS_UPDATED follows before the proximity check (Aug 55, reference 62, Feb–Mar 56 /
1765 / 56: the "superseded at m + 1" cases), so the proximity check read today's inner. Why 0 flips: 11 / 14
decisions are re-asked by the cycle's next event at m + 1 already; the other 3 are unread by MS (the cycle-0
carve-out; two crosses before their cycle is established — no POI, no CTS to confirm).

The variant tree's suite: 2172 passed, 3 skipped (the tree excludes `plans/`), 1 failed —
`test_exception_scopes::test_every_broad_except_is_on_the_allowlist` (the variant's own `except Exception` in
`_flush_fib_reasks`, a hygiene guard): no test pins the "dropped" behaviour.
