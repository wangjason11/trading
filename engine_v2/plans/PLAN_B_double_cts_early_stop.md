# Plan B — The Double-CTS Rule Becomes a True Early Stop in the first_confluence Probe

**Status:** READY TO IMPLEMENT (written 2026-09-19; every decision closed — the §2 resolution on the
`second_cts_reached` finalize moment was confirmed by the user the same day; cold-reviewed the same day
by two fresh agents — code-reference audit + implement-on-paper — and a decision-coverage pass; every
finding applied).
**Defect record:** `memory/project_sub_structure_pool_architecture.md` "PRE-EXISTING BUGS" — "Double-CTS
is classified at exit, not an early stop"; PART4 §4.4 first_confluence bullet (corrected 2026-09-19).
**Ground truth:** `memory/reference_pool_redesign_groundtruth.md` "first_confluence probe internals"
(FC(0,0) / FC(0,1): which candles the decision actually read). Reproduce with
`engine_v2/debug/probe_fc_finalize.py` (OANDA, ~2 min).
**Sequencing:** second of three — revert `9fd3143` → Plan A → **Plan B** → Plan C (PART4 §17.11).

This document is the implementation contract. A session with no memory of the design discussion can
execute it. Where it says "decided", do not re-open. Line numbers are at `e6292a3`; Plan A (which lands
first) adds lines in `market_structure.py` and `unified_probe.py`, so re-grep the anchors quoted here.

---

## 0. Prerequisites, base, sequencing

| Step | Must be true before starting |
|---|---|
| P0 | `9fd3143` reverted with its own `/commit-save` (byte-identical to `20260708_015339_c932610`, Plan A §0 P1); **Plan A landed with its own `/commit-save`**. The Plan A save is Plan B's `/compare` baseline. Ordering is load-bearing, not cosmetic: Plan B's byte-identity claim is *against the Plan A save*. Before Plan A the Phase-2 MS runs also emitted leaked tail events past `probe_end_idx`; the early stop would remove those too and the diff would mix two causes. |
| P1 | Tests green at the base (418 + Plan A's, including Plan A §5.3's **multi-cycle Phase-2 fixture** — Plan B's tests reuse it; if Plan A did not build it, build it here first, §4.0). |

Plan B is **one behavioural change** ("the first_confluence probe decides at the moment the 2nd CTS is
established") ⇒ one replay, one `/compare` (expected: byte-identical), one `/commit-save`. Do not bundle
anything from Plan C into it.

---

## 1. Today's behaviour, precisely

`unified_probe._run_phase2` (`structure/unified_probe.py:444-644`) — first_confluence only — runs, per
iteration, one MarketStructure from `current_start` **bounded at `probe_end_idx`** (`_make_market_structure(
df_probe, …, end_idx=end_idx, enforce_cts0_new_extreme=True, bos0_inner=…)`, `:502-510`; `ms.debug = True`
`:512`), then classifies on the **final** event list (`:515-616`):

```
cts_est      = every CTS_ESTABLISHED of structure 0, sorted by .idx   (_collect_cts_established :261-273)
has_rev      = any market_state == "reversal" row of structure 0      (_has_reversal :294-301)
has_rev and n_cts < 2                → reversal_in_probe   (finalize = first reversal row, :527-535)
n_cts == 0                           → end_idx_reached     (finalize = end_idx)
cycle_0_conf = first CTS_CONFIRMED with cycle_id == cycle 0's          (_collect_cts_confirmed_for_cycle :276-291)
check window = [CTS_0_est+1, cts0_anchor_idx-1] if cycle_0_conf else [CTS_0_est+1, end_idx]   (:568-575)
candidate    = most-extreme retrace in the window (df prices)          (_select_extreme_retrace_candidate)
candidate passes the 2-condition reset → advance current_start, next iteration (new MS run)
window empty / no candidate / fails  → second_cts_reached if n_cts >= 2   (finalize = cts_est[1].idx, :584)
                                        else no_retrace                    (finalize = cycle_0_conf.idx if cycle 0
                                                                            confirmed in-window, else end_idx, :585-588)
```

The **double-CTS rule** (PART4 §4.4: once a 2nd CTS is established without a qualifying retrace reset,
cycle 0 is complete and the start is confirmed) is therefore applied *after* MS has run all the way to
`probe_end_idx`. Two consequences:

- **Causally late.** The decision is knowable the moment the 2nd `CTS_ESTABLISHED` fires; the code
  reports the right *number* but only after reading up to `end_idx`. In a live/incremental model the
  sub would go live at `end_idx` rather than at the 2nd CTS — FC(0,0): 1721 instead of 1020, i.e. 701
  M15 candles late.
- **Wasted work.** `ProbeResult.iterations` is Phase-1 + Phase-2 iterations (`:759`; Phase 1 runs no MS
  and always counts ≥ 1), so the ground-truth "iter 4 / iter 2" means FC(0,0) ran **≤ 3** Phase-2 MS
  runs over `[start, 1721]` (≈ 1270 candles each) when every decision was settled by 1020, and FC(0,1)
  ran exactly **one** (to 2609, settled at 2608 — no saving there). MS is ~23 s of the replay, so the
  time saving is modest; the semantics is the reason.

**Verified (2026-09-19, live run): nothing after the 2nd `CTS_ESTABLISHED` is read by the classifier's
inputs.** FC(0,0) final iteration: `cts_est = [458, 1020, 1223, 1721]`, `cycle_0_conf = 802`,
`cts0_anchor = 784`, window `[459..783]`, candidate 630 fails → `second_cts_reached`, finalize 1020.
FC(0,1): `cts_est = [2368, 2608]`, `cycle_0_conf = 2560`, anchor 2557, window `[2369..2556]` → finalize
2608. This is general, not a coincidence of the window: a 2nd cycle can only be established after
cycle 0 is CONFIRMED (`establishing_new_cycle = cts is None or cts_phase == "CONFIRMED"`,
`market_structure.py:1408`; `cts_phase` becomes CONFIRMED only at `:1518` (pullback) / `:2009`
(proximity), and cycle-0 proximity confirmation is impossible — `cts_cycle_id > 0` gate `:1885` — so
cycle 0 has exactly one `CTS_CONFIRMED`, emitted by pullback, before cycle 1 can exist), and the check
window's upper edge is `cts0_anchor_idx - 1 < CTS_0_CONFIRMED.idx < CTS_1_ESTABLISHED.confirmed_at`
(`cts_anchor_idx = st.cts.idx` at confirm time, `:1776`, an extreme ≤ the confirm candle). So when
`n_cts >= 2` the classification reads only: `cts_est[0]`, `cts_est[1]`, cycle-0 `CTS_CONFIRMED` (+ its
`cts_anchor_idx`), and df prices in a window that ends before the 2nd CTS (MS never writes price
columns). Reversal presence matters only when `n_cts < 2` (`:520`).

**What "verified" does not cover — the event list itself.** The classifier reads the *final* event list,
and MS rewinds (`_rewind_to`, `:484-503`) wipe and rebuild it from candle 0 while ignoring earlier jump
requests. If the full run had an expiry-rewind **before** the 2nd CTS and another **after** it, the
rebuilt prefix can differ from the first-pass prefix (extra threshold updates, duplicated window events,
a moved `cycle_0_conf`), and an early-stopped run — which hands the classifier the first-pass prefix —
can then classify differently. This is the §2 exception; §5.5 says how to check for it on the reference
window (it is not expected there, but the ground-truth run cannot prove it: its `cts_est` list is itself
post-rebuild).

**A second, smaller inconsistency in the same code.** `finalize_idx` for `second_cts_reached` is
`cts_est[1].idx` (`:584`) — the 2nd CTS's **extreme** (a historical anchor, `CTS_ESTABLISHED.idx`), while
`cts0_est_idx` (`:561`) is taken from `meta["confirmed_at"]` — the **moment**. The probe's finalize is
"when the probe finalized" (real-time); the moment is the right value (PART4 §17 header principle;
§17.6 moment-not-extreme). On the reference window the two coincide for both `second_cts_reached`
cases — 1020→1020, 2608→2608 — **as read off the saved sub-build M15 streams**; the probe's own
scan-mode MS run is the same configuration from the same start (same `bos0_inner`), so the same events
are expected, but §3.2.2's print makes it a check rather than an inference.

---

## 2. Semantics (decided)

**Decided.** Phase 2's MS run **stops at the first quiescent point after the 2nd `CTS_ESTABLISHED`** of
the structure. Quiescent = no reversal watch active, no pending (scheduled) reversal, no rewind pending.
The classification then runs on the truncated event list exactly as today. Runs that never reach a 2nd
CTS (`n_cts <= 1` at `end_idx`: `no_retrace` / `end_idx_reached` / `reversal_in_probe`) are untouched —
their window is bounded by `end_idx` or by cycle 0's own anchor and they must run to the bound. Phase 1,
the main path, the main reversal probe and every sub geometry build do not use the option and are
untouched.

Why "quiescent" rather than "immediately": a reversal watch survives only while a reversal pattern is
pending with `apply <= expires_idx` (Plan A §1 L4), so at the moment of the 2nd CTS an open watch resolves
within `range_max_k` (5) candles either into a reversal (loop breaks; `n_cts >= 2` so the classification
is the same `second_cts_reached`) or into an expiry-rewind that rebuilds events from candle 0. Stopping
only once the watch is resolved means the event list handed to the classifier is never one a pending
rewind was about to rewrite. Cost: at most 5 extra candles (a watch chain — expiry → rewind → a new
close-break on the updated threshold — can extend it; do not assert "≤ 5"). This also keeps Plan B
inside LANDMINES "Probe `end_idx` Is the Supreme Bound": an inner rule may end the probe early but must
not narrow logic operating within `end_idx` — the retrace window lies entirely before the 2nd CTS, so
nothing is narrowed. After Plan A the watch expiry is clamped to `effective_end`, so a watch can never
stay open past the bound; the run ends at `end_idx` non-quiescent only if the 2nd CTS lands on the last
in-bound step with a pending reversal — then there was no early stop, and §3.2.2 must not claim one.

**Decided (user-confirmed 2026-09-19):** `finalize_idx` for `second_cts_reached` becomes
`cts_est[1].meta["confirmed_at"]` (the moment), not `cts_est[1].idx` (the extreme). Expected
byte-identical on the reference window (§1). It is the same principle Plan C applies everywhere
(`confirmed_at` for timing, `.idx` for where the extreme sits) and it is what the early stop actually keys
on — MS emits the 2nd CTS at its apply candle (`:1414-1424`, `confirmed_at = apply_idx`), which *is*
`confirmed_at`.

**Equivalence claim (what `/compare` must show):** for every first_confluence trigger, `ProbeResult`
(`starting_idx`, `finalize_idx`, `finalize_condition`, `iterations`, `bos0_inner`) is identical to the Plan A
save's. The one theoretical exception is the rebuilt-prefix case in §1. **It is not pre-authorised:** if
a `ProbeResult` differs, stop, run the §5.5 check, and ask before accepting — the early-stop value is
the causally correct one, but whether to accept a non-identical Plan B is the user's call.

---

## 3. Changes (file-level)

### 3.1 `structure/market_structure.py` — a generic, opt-in stop
1. Constructor: `stop_after_cts_established: Optional[int] = None` (keyword-only, after `bos0_inner`,
   `:298`), stored right after the `bos0_inner` block (`:341-349`) as
   `self.stop_after_cts_established = int(x) if x is not None else None`, validated `>= 1`
   (`0` would stop after the first step). `_make_market_structure` forwards `**kwargs`
   (`structure_engine.py:56-88`), so no plumbing change. Add `self.early_stop_idx: Optional[int] = None`
   (instance attribute, set by the stop; read by the probe — `ProbeResult` stays unchanged).
2. Helper:
   ```python
   def _should_stop_after_cts(self) -> bool:
       n = self.stop_after_cts_established
       if n is None:
           return False
       st = self.state
       if st.reversal_watch_active or st.pending_reversal_apply_idx is not None:
           return False                      # not quiescent — a rewind or reversal may still land
       established = sum(1 for ev in self.events if ev.type == "CTS_ESTABLISHED")
       return established >= int(n)
   ```
   Count the events (the same source of truth `_collect_cts_established` reads) rather than trusting
   `st.cts_cycle_id` — `_rewind_to` resets the state and rebuilds. One MS instance runs exactly one
   structure (`structure_id` is written only in `__init__` `:360` and chaining lives in
   `structure_engine`), so **no `structure_id` filter**: `_rewind_to` `:485` rebuilds
   `MarketStructureState` without `structure_id` (it resets to 0 — LANDMINES "Deep-Couples", point 1),
   which would make a filter on `st.structure_id` wrong after a rewind for any sid ≠ 0 caller. O(events)
   per step only while the option is set (probes; a few hundred events).
3. `run()` loop (`:422-446`): after the rewind block, at `i = next_i`:
   ```python
   i = next_i
   if self._should_stop_after_cts():
       self.early_stop_idx = int(i)
       self._dbg(f"[EARLY_STOP] i={i} cts_established>={self.stop_after_cts_established}")
       break
   ```
   The rewind branch `continue`s (`:444`) before this point, so a pending jump is always honoured first
   (the "no rewind pending" half of quiescent). Placement relative to the top-of-loop `REVERSAL` break
   (`:423`) is immaterial: a reversal applied in the same step clears watch + pending, the helper returns
   True and breaks; the full run breaks at `:423` on the next pass; both give `has_rev=True, n_cts>=2` →
   the same branch. The post-loop code (`_flush_output_arrays`, terminal reversal stamping,
   `_check_invariants_df`, Plan A's no-event-past-`effective_end` assert) runs unchanged — unprocessed
   rows keep `structure_id=-1` / `market_state=""` (`:2337-2343`), so `_has_reversal` is safe and the
   invariants (per-row) hold; quiescence guarantees no watch rows are left open.
4. Docstring of `run()`: one sentence — "With `stop_after_cts_established=N` the loop also ends at the
   first quiescent point (no reversal watch / pending reversal / pending rewind) after the N-th
   `CTS_ESTABLISHED`; `early_stop_idx` records it. Used by the first_confluence probe (Plan B)."

### 3.2 `structure/unified_probe.py`
1. `_run_phase2` `:502-510`: add `stop_after_cts_established=2` to the `_make_market_structure(...)` call.
2. After `df_probe, probe_events, _ = ms.run()`: when `ms.early_stop_idx is not None`, print one
   greppable line — `[unified_probe phase2] early stop: p2_iter=… stop_idx=… end_idx=… cts1_ext=…
   cts1_moment=…` with `cts1_ext = cts_est[1].idx` and `cts1_moment = cts_est[1].meta["confirmed_at"]`
   (so the §2 extreme-vs-moment equality is checked on the probe's own run, not inferred). **Derive
   "stopped early" from `ms.early_stop_idx`, never from `n_cts >= 2`** — a run whose 2nd CTS lands on the
   last in-bound step with a pending reversal reaches `end_idx` without stopping. `p2_iter` is the
   Phase-2-local iteration (`ProbeResult.iterations` adds Phase 1's).
3. Docstring of `_run_phase2` ("Phase 2's job is to drive MS far enough to reach CTS_0_CONFIRMED …",
   `:461-462`): add "… and it stops at the 2nd `CTS_ESTABLISHED` (Plan B) — the double-CTS rule is an
   early stop, not a classification at exit." Termination-conditions list: `second_cts_reached` → "the
   run stopped at the 2nd CTS_EST; finalize = its moment (`confirmed_at`)".

### 3.3 `structure/unified_probe.py` — the finalize moment (decided, §2)
Extract the one-liner so it is unit-testable without a fixture:
```python
def _second_cts_moment(cts_est: list) -> int:
    """finalize_idx for `second_cts_reached`: the MOMENT the 2nd CTS was established
    (meta["confirmed_at"] = its apply candle), not `.idx` (the extreme inside the pattern span)."""
    ev = cts_est[1]
    return int(ev.meta.get("confirmed_at", ev.idx))
```
`:584` → `_second_cts_fin = _second_cts_moment(cts_est) if n_cts >= 2 else None`. `ProbeResult` docstring
table (`:160`): `second_cts_reached → 2nd CTS_ESTABLISHED moment (meta["confirmed_at"])`.

### 3.4 Explicitly unchanged
`_run_phase1`, `find_true_first_breakout`, `compute_bounded_structure` (no new default),
`structure_engine` main/reversal paths, `entity_df_mutation`, `pooled_structure_build`, all detectors.
`end_idx` is still passed to Phase 2's MS — the early stop is *in addition to* the bound, never instead of
it (`n_cts <= 1` runs still need the bound).

---

## 4. Tests

### 4.0 Fixture — a multi-cycle synthetic series (shared with Plan A §5.3)
**No fixture in the suite establishes more than one CTS cycle, and no test exercises Phase 2**
(measured: `_make_uptrend_data` n=80/150 → 1 `CTS_ESTABLISHED`, never confirmed; `_make_reversing_data`
→ CTS_EST@2, CONFIRMED@34, watch@51, reversal@52, 0 rewinds; `_make_downtrend_data` → 1;
`enable_phase2=True` appears only as a mocked kwarg in `test_first_trigger_migration.py:269`). Build
`_make_multicycle_data()` in `test_bounded_structure.py`: impulse → pullback that **confirms** (a
counter-direction pattern closing back through the CTS threshold — `_seg`'s noisy pullbacks did not
confirm until idx 34, so hand-shape the pullback candles against `CandleParams`) → impulse → pullback →
impulse, and **assert in the fixture** that an unbounded MS run establishes ≥ 3 CTS cycles with cycle 0
CONFIRMED. If Plan A already built it, reuse it. Budget real time; every §4 test below depends on it.

### 4.1 MS unit — the stop option
On the multi-cycle fixture:
- `stop_after_cts_established=2` → `count(CTS_ESTABLISHED) >= 2`, `early_stop_idx is not None`, no
  `reversal` rows, `max(ev.idx) < the 3rd CTS_ESTABLISHED.confirmed_at of the unbounded run`, and the
  **ordered prefix** `full.events[:len(stopped.events)] == stopped.events` (not an idx filter — a later
  `BOS_CONFIRMED.idx`, a pullback extreme, can be `<= stop_idx`). Exact only when the unbounded run has
  no rewind after the stop point — assert that with **`REVERSAL_WATCH_START` events** (they survive
  rebuilds) or `[RV_EXPIRE]` lines via `capsys`, **not** `probe_no_break` threshold updates (wick-cross
  probes use the same reason string without a rewind, `:655-663`, and an expiry's own update is wiped by
  its rewind).
- `stop_after_cts_established=None` → byte-identical to today (events + output rows) — the guard that
  the default path is untouched.
- Quiescence: a variant where a BOS close-break with a pending reversal overlaps the 2nd CTS → the run
  does not stop while the watch is open; assert on the presence of `REVERSAL_WATCH_START` before
  `early_stop_idx` and on `early_stop_idx >= the watch's resolution candle` (expiry or apply), not on
  `max(ev.idx)` (a rewind can leave `max idx` below the expiry).

### 4.2 Probe equivalence — early stop vs classify-at-exit (`tests/test_unified_probe.py`)
On the multi-cycle fixture (and any other that reaches ≥ 2 cycles inside `end_idx`), run `_run_phase2`
twice — once as shipped, once with the stop disabled by monkeypatching the module-level name
`unified_probe._make_market_structure` (imported by name, `:70-73`) with a wrapper that pops
`stop_after_cts_established` — and assert the two `_DetResult`s are equal field by field. This is the §2
equivalence claim in unit form; it must pass before the replay.

### 4.3 Finalize = the moment (§3.3)
Unit-test `_second_cts_moment` with a stub `cts_est` whose second event has `idx != meta["confirmed_at"]`
(assert the moment is returned, and `.idx` as the fallback when the meta key is absent). A natural
fixture is hard by construction: CONFIRMED patterns confirm on a candle whose close is ≥ the pattern's
extreme (`structure_patterns.py:111-126`), so the extreme lands on the confirming candle; only a
SUCCESS pattern whose last candle's high sits below an earlier candle's wick separates them. If such a
fixture falls out of §4.0's series, add the end-to-end assert `finalize_idx == cts_est[1].meta["confirmed_at"]
!= cts_est[1].idx`; do not block on it.

### 4.4 Existing
`test_unified_probe` (all), `test_bounded_structure`, `test_pooled_structure_build`, Plan A's property
test — all green, no edits expected. Plan A's post-run assert (`max idx <= effective_end`) covers the
early-stopped run too.

---

## 5. Acceptance (`/compare` against the Plan A save)

1. **Byte-identical, every CSV, H1 and M15.** The probe outcomes are unchanged by §2's argument, and
   nothing downstream reads anything but the `ProbeResult`. Chart trace/shape counts identical.
2. `debug/probe_fc_finalize.py` before/after: the five rows identical (FC(0,0) 454 / 1020 /
   `second_cts_reached`; FC(0,1) 2365 / 2608 / `second_cts_reached`; FC(1,0), (1,1), (1,2) as after Plan A).
   The script wraps only `unified_probe` (`:81-92`) and never sees the MS run; Plan A §7 already makes it
   wrap `_make_market_structure` and keep every Phase-2 `ms` — extend that to print `ms.early_stop_idx`
   per iteration and the `cts1_ext` / `cts1_moment` pair. Expected: `stop_idx` a few candles after 1020
   for FC(0,0)'s last iteration and after 2608 for FC(0,1); `None` for the `n_cts <= 1` rows;
   `cts1_ext == cts1_moment` for both (if not, the §2 finalize changes the number — stop and ask).
3. Replay log: the new lines are `[unified_probe phase2] early stop …` — at least 2 (FC(0,0)'s last
   iteration and FC(0,1)'s only one) and at most 4 (FC(0,0)'s earlier reset iterations stop early only if
   they also established a 2nd CTS before `end_idx`; record the count) — plus the `[EARLY_STOP]` `_dbg`
   line (`ms.debug = True` prints it) and **fewer** `[POST_STEP]` lines after each stop. Criterion: no new
   `warning|skipping|unavailable|degenerate|pending|no sid` lines.
4. `=== Replay Timing ===` displayed; the MS bucket may drop by a fraction of a second. Not a criterion.
5. **Any CSV delta is a regression — stop and ask.** The only mechanism that could produce one is the
   rebuilt-prefix case (§1/§2). To check it: in the Plan A save's `run.log`, per first_confluence Phase-2
   iteration (delimited by the `[unified_probe …]` prints; all `_dbg` lines are on), look for a
   `[REWIND] … jump_to=J` with `J-1 < stop_idx` **and** one with `J-1 > stop_idx` in the same iteration.
   Neither → the claim holds by determinism and the delta is something else (a real regression). Both →
   name it, show the two rewinds, and let the user decide whether the new (causal) value is accepted.
6. No chart change is expected; a short chart look is still the habit — then `/commit-save`.

---

## 6. Docs in the same commit
- **PART4 §4.4** first_confluence bullet (`:568-584`): "Note the double-CTS rule is currently a
  *classification at exit* … Plan B makes it a true early stop …" (`:577-579`) → "The double-CTS rule is
  an **early stop**: Phase-2 MS stops at the first quiescent point after the 2nd `CTS_ESTABLISHED`
  (Plan B, commit …); finalize = that CTS's moment (`confirmed_at`)". `:575` "(finalize = the 2nd
  `CTS_ESTABLISHED.idx`, native M15)" → "(finalize = the 2nd `CTS_ESTABLISHED`'s `confirmed_at`, native M15)".
- **PART4 §5** (`:819-820`) "Phase-2 `second_cts_reached` → 2nd `CTS_ESTABLISHED` idx" → "moment
  (`confirmed_at`)". **§17.11** Plan B line (`:2520-2526`) → landed, commit.
- **GOTCHAS** finalize table (`:1592`): `second_cts_reached | 2nd CTS_ESTABLISHED moment (meta["confirmed_at"]) …`
  (drop the "Plan B changes this" parenthetical).
- **`ProbeResult` docstring** (§3.3) and `_run_phase2` docstring (§3.2).
- **LANDMINES** "Probe `end_idx` Is the Supreme Bound" (`:383-391`): add one sentence — "Plan B's early
  stop at the 2nd CTS_EST is the sanctioned form of 'stop after 2 CTS_EST': it ends the run but the
  retrace window it classifies lies before the 2nd CTS, so nothing inside `end_idx` is narrowed."
- **Memory:** `project_sub_structure_pool_architecture.md` PRE-EXISTING → Plan B FIXED (commit);
  `reference_pool_redesign_groundtruth.md`: note the stop idx per cycle; `MEMORY.md` status.

---

## 7. Out of scope (do not touch here)
- Phase 1 (deterministic) already stops per iteration at the retrace decision; nothing to gain.
- `n_cts <= 1` runs must reach `end_idx` — with cycle 0 unconfirmed the retrace window is
  `[CTS_0_EST+1, end_idx]`, with cycle 0 confirmed it is `[CTS_0_EST+1, cts0_anchor-1]` but a 2nd CTS may
  still arrive before `end_idx`; an earlier stop there would be the very bypass LANDMINES forbids.
- The probe cache (Plan C §17.8) and the record's `probe_finalize_idx` semantics (Plan C §17.4) — Plan C.
- A live/incremental probe (candle-by-candle) — Phase 3.
- `_rewind_to` replaying from 0 / ignoring earlier jumps / dropping `structure_id` (Plan A §8,
  LANDMINES "Deep-Couples") — separate.

## 8. Definition of done
§4 tests green (4.2 equivalence and 4.3 moment are the two that matter; 4.0's fixture asserts its own
cycle count); §5.1 byte-identical `/compare` against the Plan A save (any delta → §5.5, user decides);
§6 docs in the same commit; `/commit-save`; memory status updated; then Plan C.
