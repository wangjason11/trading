# PLAN_X — the item-10 picks: the REVERSAL_CANDIDATE pattern fix + the MS risk census (Week 9 Part 1)

**Status: X1 BUILT = its prediction (§X1 "Result") · X2 PLANNED (method below).** Register `IDEA_PARKING_LOT.md` §D SESSION
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
