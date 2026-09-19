# Plan B — The Double-CTS Rule Becomes a True Early Stop in the first_confluence Probe

**Status:** READY TO IMPLEMENT (written 2026-09-19). One author resolution is flagged in §2 (the
finalize idx of `second_cts_reached` moves from the 2nd CTS's *extreme* to its *moment*; byte-identical
on the reference window) — confirm before coding.
**Defect record:** `memory/project_sub_structure_pool_architecture.md` "PRE-EXISTING BUGS" — "Double-CTS
is classified at exit, not an early stop"; PART4 §4.4 first_confluence bullet (corrected 2026-09-19).
**Ground truth:** `memory/reference_pool_redesign_groundtruth.md` "first_confluence probe internals"
(FC(0,0) iter 4 / FC(0,1) iter 2: which candles the decision actually read). Reproduce with
`engine_v2/debug/probe_fc_finalize.py` (OANDA, ~2 min).
**Sequencing:** second of three — revert `9fd3143` → Plan A → **Plan B** → Plan C (PART4 §17.11).

This document is the implementation contract. A session with no memory of the design discussion can
execute it. Where it says "decided", do not re-open.

---

## 0. Prerequisites, base, sequencing

| Step | Must be true before starting |
|---|---|
| P0 | `9fd3143` reverted; **Plan A landed with its own `/commit-save`**. That save is Plan B's `/compare` baseline. Ordering is load-bearing, not cosmetic: Plan B's byte-identity claim is *against the Plan A save*. Before Plan A the Phase-2 MS runs also emitted leaked tail events past `probe_end_idx`; the early stop would remove those too and the diff would mix two causes. |
| P1 | Tests green at the base (418 + Plan A's). |

Plan B is **one behavioural change** ("the first_confluence probe decides at the moment the 2nd CTS is
established") ⇒ one replay, one `/compare` (expected: byte-identical), one `/commit-save`. Do not bundle
anything from Plan C into it.

---

## 1. Today's behaviour, precisely

`unified_probe._run_phase2` (`structure/unified_probe.py:~451-620`) — first_confluence only — runs, per
iteration, one MarketStructure from `current_start` **bounded at `probe_end_idx`** (`_make_market_structure(
df_probe, …, end_idx=end_idx, enforce_cts0_new_extreme=True, bos0_inner=…)`, `:502-511`), then classifies
on the **final** event list:

```
cts_est      = every CTS_ESTABLISHED of structure 0          (_collect_cts_established)
has_rev      = any market_state == "reversal" row             (_has_reversal)
has_rev and n_cts < 2                → reversal_in_probe   (finalize = first reversal row)
n_cts == 0                           → end_idx_reached     (finalize = end_idx)
cycle_0_conf = CTS_CONFIRMED of cycle 0                       (_collect_cts_confirmed_for_cycle)
check window = [CTS_0_est+1, cts0_anchor_idx-1] if cycle_0_conf else [CTS_0_est+1, end_idx]
candidate    = most-extreme retrace in the window (df prices)  (_select_extreme_retrace_candidate)
candidate passes the 2-condition reset → advance current_start, next iteration (new MS run)
window empty / no candidate / fails  → second_cts_reached if n_cts >= 2   (finalize = cts_est[1].idx)
                                        else no_retrace                    (finalize = cycle_0_conf.idx, else end_idx)
```

The **double-CTS rule** (PART4 §4.4: once a 2nd CTS is established without a qualifying retrace reset,
cycle 0 is complete and the start is confirmed) is therefore applied *after* MS has run all the way to
`probe_end_idx`. Two consequences:

- **Causally late.** The decision is knowable the moment the 2nd `CTS_ESTABLISHED` fires; the code
  reports the right *number* but only after reading up to `end_idx`. In a live/incremental model the
  sub would go live at `end_idx` rather than at the 2nd CTS — FC(0,0): 1721 instead of 1020, i.e. 701
  M15 candles late.
- **Wasted work.** FC(0,0) runs 4 iterations of MS over `[start, 1721]` ≈ 1270 candles each when every
  decision was settled by 1020; FC(0,1) 2 iterations to 2609 settled at 2608 (no saving there). MS is
  ~23 s of the replay, so the time saving is modest; the semantics is the reason.

**Verified (2026-09-19, live run): nothing after the 2nd `CTS_ESTABLISHED` is read by the decision.**
FC(0,0) iter 4: `cts_est = [458, 1020, 1223, 1721]`, `cycle_0_conf = 802`, `cts0_anchor = 784`, window
`[459..783]`, candidate 630 fails → `second_cts_reached`, finalize 1020. FC(0,1) iter 2: `cts_est =
[2368, 2608]`, `cycle_0_conf = 2560`, anchor 2557, window `[2369..2556]` → finalize 2608. In both, the
window and every input to the decision lie before the 2nd CTS. This is general, not a coincidence of
the window: a 2nd cycle can only be established after cycle 0 is CONFIRMED (`establishing_new_cycle =
cts is None or cts_phase == "CONFIRMED"`, `market_structure.py:~1408`), and the check window's upper
edge is `cts0_anchor_idx - 1 < CTS_0_CONFIRMED.idx < CTS_1_ESTABLISHED`. So when `n_cts >= 2` the
classification reads only: `cts_est[0]`, `cts_est[1]`, cycle-0 `CTS_CONFIRMED` (+ its `cts_anchor_idx`),
and df prices in a window that ends before the 2nd CTS. Reversal presence matters only when
`n_cts < 2`.

**A second, smaller inconsistency in the same code.** `finalize_idx` for `second_cts_reached` is
`cts_est[1].idx` — the 2nd CTS's **extreme** (a historical anchor, `CTS_ESTABLISHED.idx`), while three
lines above `cts0_est_idx` is taken from `meta["confirmed_at"]` — the **moment**. The probe's finalize
is "when the probe finalized" (real-time); the moment is the right value (PART4 §17 header principle;
§17.6 moment-not-extreme). On the reference window the two coincide for both `second_cts_reached`
cases (1020→1020, 2608→2608 in the saved M15 streams; the extreme≠moment pairs 1223/1224 and
2828/2829 are not finalize values), so fixing it here is byte-identical.

---

## 2. Semantics (decided) — and one author resolution

**Decided.** Phase 2's MS run **stops at the first quiescent point after the 2nd `CTS_ESTABLISHED`** of
structure 0. Quiescent = no reversal watch active, no pending (scheduled) reversal, no rewind pending.
The classification then runs on the truncated event list exactly as today. `no_retrace` runs
(`n_cts == 1` at `end_idx`) are untouched — their window *is* `end_idx`. Phase 1, the main path, the
main reversal probe and every sub geometry build do not use the option and are untouched.

Why "quiescent" rather than "immediately": a reversal watch open at the moment of the 2nd CTS resolves
within `range_max_k` (5) candles either into a reversal (loop breaks; `n_cts >= 2` so the classification
is the same `second_cts_reached`) or into an expiry-rewind that rebuilds events from the watch anchor.
Stopping only once the watch is resolved means the event list handed to the classifier is never one a
pending rewind was about to rewrite. Cost: at most 5 extra candles. This also keeps Plan B inside
LANDMINES "Probe `end_idx` Is the Supreme Bound": an inner rule may end the probe early but must not
narrow logic operating within `end_idx` — the retrace window lies entirely before the 2nd CTS, so
nothing is narrowed.

**AUTHOR'S RESOLUTION — confirm:** `finalize_idx` for `second_cts_reached` becomes
`cts_est[1].meta["confirmed_at"]` (the moment), not `cts_est[1].idx` (the extreme). Byte-identical on the
reference window (§1). It is the same principle Plan C applies everywhere (`confirmed_at` for timing,
`.idx` for where the extreme sits) and it is what the early stop actually keys on — MS emits the 2nd
CTS at its apply candle, which *is* `confirmed_at`. If you prefer to keep `.idx` for now, drop §3.3's
one line; nothing else depends on it.

**Equivalence claim (what `/compare` must show):** for every first_confluence trigger, `ProbeResult`
(`starting_idx`, `finalize_idx`, `finalize_condition`, `iterations`, `bos0_inner`) is identical to the Plan A
save's. The one theoretical exception: a rewind *after* the stop point that would have rebuilt a
different prefix (Plan A §8 — `_rewind_to` replays from 0 ignoring earlier jumps). If that ever shows up
as a delta, the early-stop value is the causally correct one and the delta is documented, not reverted —
but it is not expected on this window (the ground-truth run lists the final `cts_est` sequences and they
contain the stop-point CTS at the same idx).

---

## 3. Changes (file-level)

### 3.1 `structure/market_structure.py` — a generic, opt-in stop
1. Constructor: `stop_after_cts_established: Optional[int] = None` (keyword-only, after `bos0_inner`),
   stored as `self.stop_after_cts_established`. Default `None` = today's behaviour for every existing
   caller (`_make_market_structure` forwards `**kwargs`, so no plumbing change there).
2. Helper:
   ```python
   def _should_stop_after_cts(self) -> bool:
       n = self.stop_after_cts_established
       if n is None:
           return False
       st = self.state
       if st.reversal_watch_active or st.pending_reversal_apply_idx is not None:
           return False                      # not quiescent — a rewind or reversal may still land
       established = sum(
           1 for ev in self.events
           if ev.type == "CTS_ESTABLISHED"
           and int(ev.meta.get("structure_id", -1)) == int(st.structure_id)
       )
       return established >= int(n)
   ```
   Count the events (the same source of truth `_collect_cts_established` reads) rather than trusting
   `st.cts_cycle_id` — the counter survives rewinds only because the rebuild recreates it, and counting
   is O(events) per step only while the option is set (probes; a few hundred events).
3. `run()` loop (`:422-446`): after the rewind block, at `i = next_i`:
   ```python
   i = next_i
   if self._should_stop_after_cts():
       self._dbg(f"[EARLY_STOP] i={i} cts_established>={self.stop_after_cts_established}")
       break
   ```
   The rewind branch `continue`s before this point, so a pending jump is always honoured first (that is
   the "no rewind pending" half of quiescent). The post-loop code (`_flush_output_arrays`, levels,
   terminal reversal stamping, invariants, Plan A's no-event-past-`effective_end` assert) runs unchanged.
4. Docstring of `run()`: one sentence — "With `stop_after_cts_established=N` the loop also ends at the
   first quiescent point (no reversal watch / pending reversal / pending rewind) after the N-th
   `CTS_ESTABLISHED` of this structure; used by the first_confluence probe (Plan B)."

### 3.2 `structure/unified_probe.py`
1. `_run_phase2` `:502-511`: add `stop_after_cts_established=2` to the `_make_market_structure(...)` call.
2. After `df_probe, probe_events, _ = ms.run()`: log one greppable line when the stop fired —
   `[unified_probe phase2] early stop: iter=… n_cts=… stop_idx=… end_idx=…` where `stop_idx = max(ev.idx)`.
   Derive "fired" from `len(cts_est) >= 2` (the only way the run ends before `end_idx` without a
   reversal); do not add MS state to the result for this.
3. Docstring of `_run_phase2` ("Phase 2's job is to drive MS far enough to reach CTS_0_CONFIRMED …"):
   add "… and it stops at the 2nd `CTS_ESTABLISHED` (Plan B) — the double-CTS rule is an early stop, not
   a classification at exit." Termination-conditions list: `second_cts_reached` → "the run stopped at
   the 2nd CTS_EST; finalize = its moment (`confirmed_at`)".

### 3.3 `structure/unified_probe.py` — the finalize moment (AUTHOR'S RESOLUTION, §2)
`_second_cts_fin = int(cts_est[1].meta.get("confirmed_at", cts_est[1].idx)) if n_cts >= 2 else None`
(mirrors the `cts0_est_idx` line). `ProbeResult` docstring table: `second_cts_reached → 2nd
CTS_ESTABLISHED moment (meta["confirmed_at"])`.

### 3.4 Explicitly unchanged
`_run_phase1`, `find_true_first_breakout`, `compute_bounded_structure` (no new default), `structure_engine`
main/reversal paths, `entity_df_mutation`, `pooled_structure_build`, all detectors. `end_idx` is still
passed to Phase 2's MS — the early stop is *in addition to* the bound, never instead of it (`no_retrace`
runs still need the bound).

---

## 4. Tests

### 4.1 MS unit — the stop option (`tests/test_market_structure_early_stop.py`, or inside the existing MS test module)
On the `_make_reversing_data` / `_make_uptrend_data` fixtures (`tests/test_bounded_structure.py`), pick
one that establishes ≥ 3 CTS cycles unbounded (assert it does — otherwise the test proves nothing):
- `stop_after_cts_established=2` → the run ends with `count(CTS_ESTABLISHED) >= 2`, `max(ev.idx) <
  the 3rd CTS_ESTABLISHED.confirmed_at of the unbounded run`, no `reversal` rows, and **the event list
  equals the unbounded run's events with `idx <= stop_idx`** (prefix-equivalence at the stop point —
  exact on a fixture with no rewind after it; assert the fixture has none: no `probe_no_break`
  threshold update after `stop_idx` in the unbounded run).
- `stop_after_cts_established=None` → byte-identical to today (events + output rows) — the guard that
  the default path is untouched.
- Quiescence: a fixture where a BOS close-break opens a reversal watch in the same 5 candles as the 2nd
  CTS_EST → the run does not stop until the watch expires (a `probe_no_break` update is present) or
  reverses; assert `max(ev.idx) >= watch expiry idx`.

### 4.2 Probe equivalence — early stop vs classify-at-exit (`tests/test_unified_probe.py`)
For the Phase-2 fixtures already in the module plus one with ≥ 2 cycles inside `end_idx`: run
`_run_phase2` twice — once as shipped, once with the stop disabled (monkeypatch
`unified_probe._make_market_structure` to drop the kwarg, or expose a private `_stop_after_cts`
parameter defaulting to 2) — and assert the two `_DetResult`s are equal field by field. This is the
§2 equivalence claim in unit form; it must pass before the replay.

### 4.3 Finalize = the moment (§3.3)
A fixture whose 2nd CTS's extreme precedes its apply candle (a `continuous` 3-candle breakout: extreme
on the first candle, apply on the third; or any 2-candle pattern that needs confirmation) — assert
`finalize_idx == cts_est[1].meta["confirmed_at"] != cts_est[1].idx`. Without such a fixture the change is
untested by construction (the reference window can't distinguish them).

### 4.4 Existing
`test_unified_probe` (all), `test_bounded_structure`, `test_pooled_structure_build`, Plan A's property
test — all green, no edits expected. Plan A's post-run assert (`max idx <= effective_end`) covers the
early-stopped run too.

---

## 5. Acceptance (`/compare` against the Plan A save)

1. **Byte-identical, every CSV, H1 and M15.** The probe outcomes are unchanged by §2's argument, and
   nothing downstream reads anything but the `ProbeResult`. Chart trace/shape counts identical.
2. `debug/probe_fc_finalize.py` before/after: the five rows identical (FC(0,0) 454 / 1020 /
   `second_cts_reached` / iter 4; FC(0,1) 2365 / 2608 / `second_cts_reached` / iter 2; FC(1,0), (1,1), (1,2)
   as after Plan A). Add a printed `stop_idx` (max event idx of the last iteration's MS run) so the stop
   is visible: expected 1020 + ≤5 for (0,0), 2608 + ≤5 for (0,1), `end_idx` for the three `no_retrace` rows.
3. Replay log: the only new lines are `[unified_probe phase2] early stop …` — at least 2 (FC(0,0)
   iter 4 and FC(0,1) iter 2, the two ground-truth runs with `n_cts >= 2`) and at most 6 (the earlier
   reset iterations stop early only if their run also established a 2nd CTS before `end_idx`; record
   the count); no new `warning|skipping|unavailable|degenerate|pending|no sid` lines.
4. `=== Replay Timing ===` displayed; the MS bucket may drop by a fraction of a second. Not a criterion.
5. **Any CSV delta is a regression** — with one named exception: a first_confluence `ProbeResult` that
   differs because a rewind after the stop point rebuilt a different prefix in the old full run (§2).
   Show the `[REWIND]`/`[RV_EXPIRE]` lines from the old run past the stop idx to claim it; then the new
   value is the intended one and the ground-truth memory table is updated with it.
6. No chart change is expected; a short chart look is still the habit — then `/commit-save`.

---

## 6. Docs in the same commit
- **PART4 §4.4** first_confluence bullet (`:568-582`): "Note the double-CTS rule is currently a
  *classification at exit* … Plan B makes it a true early stop" → "The double-CTS rule is an **early
  stop**: Phase-2 MS stops at the first quiescent point after the 2nd `CTS_ESTABLISHED` (Plan B, commit …);
  finalize = that CTS's moment (`confirmed_at`)". `:574-575` "(finalize = the 2nd `CTS_ESTABLISHED.idx`,
  native M15)" → "(finalize = the 2nd `CTS_ESTABLISHED`'s `confirmed_at`, native M15)".
- **PART4 §5** (`:817-818`) "Phase-2 `second_cts_reached` → 2nd `CTS_ESTABLISHED` idx" → "moment
  (`confirmed_at`)". **§17.11** Plan B line → landed, commit.
- **GOTCHAS** finalize table (`:1592`): `second_cts_reached | 2nd CTS_ESTABLISHED moment (meta["confirmed_at"]) …`.
- **`ProbeResult` docstring** (§3.3) and `_run_phase2` docstring (§3.2).
- **LANDMINES** "Probe `end_idx` Is the Supreme Bound": add one sentence — "Plan B's early stop at the
  2nd CTS_EST is the sanctioned form of 'stop after 2 CTS_EST': it ends the run but the retrace window
  it classifies lies before the 2nd CTS, so nothing inside `end_idx` is narrowed."
- **Memory:** `project_sub_structure_pool_architecture.md` PRE-EXISTING → Plan B FIXED (commit);
  `reference_pool_redesign_groundtruth.md`: note the stop idx per cycle; `MEMORY.md` status.

---

## 7. Out of scope (do not touch here)
- Phase 1 (deterministic) already stops per iteration at the retrace decision; nothing to gain.
- `no_retrace` runs must reach `end_idx` — the retrace window is `[CTS_0_EST+1, end_idx]` when cycle 0
  does not confirm inside the window; an earlier stop there would be the very bypass LANDMINES forbids.
- The probe cache (Plan C §17.8) and the record's `probe_finalize_idx` semantics (Plan C §17.4) — Plan C.
- A live/incremental probe (candle-by-candle) — Phase 3.
- `_rewind_to` replaying from 0 / ignoring earlier jumps (Plan A §8) — separate.

## 8. Definition of done
§4 tests green (4.2 equivalence and 4.3 moment fixture are the two that matter); §5.1 byte-identical
`/compare` against the Plan A save (or the one named exception, documented); §6 docs in the same commit;
`/commit-save`; memory status updated; then Plan C.
