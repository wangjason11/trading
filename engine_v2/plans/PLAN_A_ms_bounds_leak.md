# Plan A — Bounded MarketStructure Runs Read Nothing Past Their Bound

**Status:** READY TO IMPLEMENT (written 2026-09-19). One author judgement call is flagged in §2 —
confirm it before coding, it is a one-line difference.
**Defect record:** `LANDMINES.md` "MarketStructure Range Look-Ahead Leaks Past `end_idx`" (corrected by
this plan — the original entry named only one of the two emission paths).
**Ground truth:** `memory/reference_pool_redesign_groundtruth.md` (the first_confluence probe table;
"Observed leaks past `end_idx`"). Reproduce with `engine_v2/debug/probe_fc_finalize.py` (OANDA, ~2 min).
**Sequencing:** first of three — `git revert 9fd3143` → **Plan A** → Plan B → Plan C (PART4 §17.11).

This document is the implementation contract. A session with no memory of the design discussion can
execute it. Where it says "decided", do not re-open.

---

## 0. Prerequisites, base, sequencing

| Step | Must be true before starting |
|---|---|
| P0 | `9fd3143` (Stage 3.2b) has been **`git revert`ed** on `week8-volmom-multitf` (kept in history). Tree = Stage 3.2a semantics. |
| P1 | The revert has its own **`/commit-save`** (78 s) and its `/compare` against `20260708_015339_c932610` is **byte-identical on every CSV** (the revert restores 3.2a exactly; if not, stop — the revert is wrong). This save is Plan A's `/compare` baseline. |
| P2 | 418+ tests green at the base. |

Plan A is **one behavioural change** ("a bounded MS run reads only candles ≤ its bound") ⇒ one replay,
one `/compare`, one chart-review pause, one `/commit-save`. Do not bundle Plan B into it.

---

## 1. The defect, precisely

A bounded run is `MarketStructure(end_idx=B)` (via `compute_bounded_structure(end_idx=B)`). Its main loop
is bounded (`effective_end = min(n-1, B)`, `while i <= effective_end`, `market_structure.py:414-422`), but
four things inside the loop still see the dataframe end `n-1`, not `B`:

| # | Site | What leaks | Observed on the reference window |
|---|---|---|---|
| L1 | `_step_anchor` `:897` and `_post_apply_range_check` `:1535`: `D = min(i + self.range_max_k, n - 1)` | `D` is the horizon for **pattern applies** (`apply <= D`, `:1015/1040/1060`) and for the range **back-fill** (`for k in range(i, min_d)`, `:952/1549`). A pattern anchored at `i <= B` applies at `apply <= i+5`, possibly `> B` — `_apply_pattern_at_apply_idx` fires there, and the back-fill steps `_replay_step_no_patterns(k)` run for `k > B` (per-candle CTS updates, proximity confirmation, BOS probes). | FC(1,0): `CTS_CONFIRMED@2844` on a run bounded at 2843 — the **only** reason its `finalize_idx` is 2844 and its retrace window `[2761..2841]` instead of `[2761..2843]`. FC(0,0): events to 1725 on a bound of 1721. |
| L2 | `_finalize_range_candidate_offline` `:1127-1144`: `RANGE_STARTED` and `STATE_CHANGED→RANGE` are stamped at **`confirm_idx`** = the pre-computed `is_range_confirm_idx` label (`range_label.py`: the first close in `[i+2, i+5]` inside candle `i`'s range), which is read raw from the df and **never compared to the bound**. | A range candidate at `i <= B` whose confirming close is at `confirm_idx > B` is activated and its events stamped in the future. **Clamping `D` does not touch this path** — the LANDMINES entry's "clamp `D`" is necessary but not sufficient. | FC(1,1): `RANGE_STARTED` / `STATE_CHANGED@3050` on a run bounded at 3047. |
| L3 | `BreakoutPatterns` guards on `len(self.df)` (`structure_patterns.py:115/156/181/349/439/529`) | The detector reads candles past `B` to decide SUCCESS / CONFIRMED. MS drops a candidate whose apply is `> D`, which is *usually* equivalent to "not knowable yet" — but `detect_best_for_anchor` returns ONE pattern per anchor by **priority** (`continuous` SUCCESS at `idx+2` pre-empts a 2-candle SUCCESS at `idx+1`). At `B = idx+1`: full frame → `continuous` wins, is dropped (`idx+2 > D`), **no pattern**; a frame that ends at `B` → `continuous` is `None`, the 2-candle pattern is returned and applied. A future candle changed the decision. | Not measured (needs a 3-candle and a 2-candle pattern on the same anchor exactly at a bound); real by code reading. |
| L4 | `_start_reversal_watch` `:710`: `reversal_watch_expires_idx = min(i + range_max_k, len(self.df) - 1)`; `_maybe_expire_reversal_watch` `:784`: `jump_to = min(anchor + 1, len(self.df) - 1)`; `_rewind_to` `:482` clamps to `n-1`. | A watch opened within 5 candles of `B` expires past `B` in the bounded run but **at the edge** in a run whose frame ends at `B` (today's data-edge behaviour: the watch expires on the last candle → false-break → rewind). The bounded run and the truncated run disagree at the tail. | Not measured; see §2 (judgement call). |

**Why it matters (unchanged from LANDMINES):** a bounded run is not reproducible from its stated bound;
`unified_probe`'s docstring ("Inclusive supreme upper bound", `unified_probe.py:691`) and
`compute_bounded_structure`'s ("The run never writes/emits past it", `structure_engine.py:~870`) are
false today; the probe's `finalize_idx` — and through the retrace window, `starting_idx` = the pool
key — can depend on candles the probe was told not to use.

**Who is actually affected (verified 2026-09-19).** Every sub geometry build slices its frame to the
bound first — `_build_or_get_sub_geometry` takes `entity_df.iloc[slice_begin:run_cap_abs + 1]`
(`entity_df_mutation.py:1076`) and `build_one_sid` builds through it (`:1219`) — so there
`n-1 == effective_end` and **Plan A is a no-op by construction**. Main runs pass `end_idx=None`. The only production paths that run MS on a frame longer
than the bound are:
- `unified_probe._run_phase2` (`df_probe = df.copy()` of the whole M15 entity frame, `end_idx = probe_end_idx`)
  — the first_confluence probe. **This is where every observed leak came from.**
- `unified_probe._run_phase1` / `find_true_first_breakout` (all five trigger types + main reversals) —
  L3 only, and provably equivalent already (§4), so no output change is expected there.
- `pooled_structure_build.build_structure_geometry` — test-only twin, deleted by Plan C.

So the expected replay footprint is small (§6). The fix is still made **inside MS and the detector**,
not at the callers, so that every present and future bounded run has the same semantics.

---

## 2. Semantics (decided) — and one judgement call

**Definition.** A bounded run `MarketStructure(df, start_idx=s, end_idx=B).run()` produces exactly the
events and output rows of `MarketStructure(df_B, start_idx=s, end_idx=None).run()` where `df_B` is the same
frame **truncated to `[0, B]`** (with its look-ahead feature labels recomputed on the truncated frame).
`effective_end` **is the run's data edge**: nothing past it is read, and everything the run does at
the edge is what it already does at the real data edge today.

**What this is NOT.** It is not "clip of the natural-end run ≡ bounded run" (prefix-equivalence). The
two can differ in the last `range_max_k` (5) candles before `B`, inherently: the natural-end run may be
back-filling a pattern that applies past `B` (rows written with `freeze_range=True`, next anchor
`apply+1`), while the bounded run processes those candles as anchors (unfrozen range expansion,
range candidates, a pattern the natural run's priority rule pre-empted); a watch open at `B` expires
in one run and not the other; and an extra rewind rebuilds from 0 ignoring earlier jump requests
(`_rewind_to`, `:489-503`), which can change events before the rewind point. This is the 5-candle
pending-confirmation nature of MS, not a bug, and it is exactly why the pool (Plan C) runs geometry to
the **data edge** and clips *windows* out of one run rather than relying on bounded builds. Do not try
to make prefix-equivalence hold; do not "look ahead but suppress emission" (rejected in LANDMINES —
it keeps future information inside the state machine).

**Judgement call (AUTHOR'S RESOLUTION — confirm before coding): L4, the reversal watch at the bound.**
Two consistent choices:
- **(a) Truncation semantics for the watch too (recommended, adopted below).** `expires_idx =
  min(i + range_max_k, effective_end)`. A watch open at `B` expires at `B` (false break → BOS threshold
  to the anchor wick → rewind to `anchor+1`), exactly as at the real data edge today. The definition
  above holds unconditionally, the property test (§5.1) is unconditional, and there is one semantics
  for "edge of what the run can see".
- (b) Leave the watch open past the bound (`min(i + range_max_k, n-1)` unchanged). Nothing past `B` is
  read either way (the reversal pattern's apply is already bounded by `D`), but the bounded run then
  differs from the truncated run whenever a watch straddles the bound, the property test must special-
  case it, and the data edge behaves differently from a bound.

(a) is a behavioural change only for probe Phase-2 runs with a watch open at `probe_end_idx`. On the
reference window this is not known to occur; §6 says how to detect it if it does.

---

## 3. Changes (file-level)

### 3.1 `structure/market_structure.py`
1. **`__init__`** (after `self.end_idx = …`, `:324`, and before `self._bp = BreakoutPatterns(self.df)`, `:364`):
   ```python
   n = len(self.df)
   self._effective_end = n - 1 if self.end_idx is None else min(n - 1, int(self.end_idx))
   ```
   `run()` (`:413-416`) and `_get_cts0_tfb` (`:1718-1721`) use `self._effective_end` instead of
   recomputing it.
2. **`BreakoutPatterns(self.df, end_idx=self._effective_end)`** (`:364`).
3. **L1:** `:897` and `:1535` → `D = min(i + self.range_max_k, self._effective_end)`. The `n = len(self.df)`
   locals become unused — delete them.
4. **L2:** `_is_range_candle_given_confirm` (`:1092-1094`):
   ```python
   confirm_idx = int(self.df.iloc[i].get("is_range_confirm_idx", -1))
   if confirm_idx < 0 or confirm_idx > self._effective_end:
       return (False, None)          # the confirming close has not happened yet
   ```
   This is the whole L2 fix: both call sites (`:948`, `:1539`) go through it, and
   `_finalize_range_candidate_offline` re-reads the label only after `range_confirmed` was True.
   Equivalent to recomputing `apply_is_range_labels` on the truncated frame (the label locks the FIRST
   confirming close; if that close is `> B` no close `<= B` confirmed it).
5. **L4 (per §2(a)):** `:710` → `min(int(i) + int(self.range_max_k), self._effective_end)`; `:784` →
   `min(anchor + 1, self._effective_end)`; `_rewind_to` `:481-482` → clamp to `self._effective_end`.
6. **Post-run assert** at the end of `run()` (unconditional — O(events), and it is the invariant this
   plan exists for):
   ```python
   if self.events:
       _max_idx = max(int(ev.idx) for ev in self.events)
       assert _max_idx <= self._effective_end, (
           f"[market_structure] event past effective_end: max idx {_max_idx} > {self._effective_end}")
   ```
   Every event type's `idx` is ≤ its apply candle (`BOS_CONFIRMED.idx` and `CTS_ESTABLISHED.idx` are
   extremes *inside* the pattern span; `RANGE_STARTED.idx = confirm_idx`; threshold updates at `i`),
   so after L1+L2+L4 this cannot fire. If it fires, a forward read was missed — do not weaken it.
7. **`run()` docstring**: replace "If end_idx is set, processing stops after that idx (inclusive)" with
   the §2 definition (two sentences: truncation semantics; not prefix-equivalence).

Nothing else in MS changes. In particular the main-path behaviour (`end_idx=None`) is untouched by
construction: every edit is `n-1 → self._effective_end`, and `self._effective_end == n-1` there.

### 3.2 `patterns/structure_patterns.py`
```python
def __init__(self, df, end_idx: Optional[int] = None):
    self.df = df
    # Visible length: a bounded caller (MarketStructure / unified_probe with end_idx) hands the
    # inclusive last candle it may read; the detector treats it exactly like the end of the frame.
    self.n_visible = len(df) if end_idx is None else min(len(df), int(end_idx) + 1)
```
Replace the six length guards — `:115` (`_price_confirmation`, `k >= len(df)`), `:156`
(`_price_confirmation_1step`), `:181` (`continuous`, `idx + 2 >= len(self.df)`), `:349/:439/:529`
(the three 2-candle patterns, `idx + 1 >= len(self.df)`) — with `self.n_visible`. `confirmation_threshold`
(`:107-108`) reads `i+1` but is only reached from a 2-candle pattern that already passed its guard.
`_row` is untouched. Add one docstring line to `detect_best_for_anchor` under its "Notes": "With
`end_idx`, candles past it do not exist for this detector: a SUCCESS/CONFIRMED that would need them is
reported as `None` / unconfirmed, never as a candidate to be dropped later."
`pattern_engine.py:49` (offline `pat*` marker columns over the whole frame) keeps `BreakoutPatterns(df2)`
unbounded — MS never reads the `pat*` columns (verified: its df reads are `is_range*`, `candle_type`,
`direction`, o/h/l/c and its own output columns).

### 3.3 `structure/unified_probe.py`
- `:715` → `bp = BreakoutPatterns(df, end_idx=end_idx)`. Phase 1 is then bounded by construction rather
  than by the `est > hi` drop in `find_true_first_breakout` (§4 — equivalent; expect no change).
- Phase 2 (`_run_phase2`, after `df_probe, probe_events, _ = ms.run()`): the MS assert (§3.1.6) already
  covers it; add nothing but a comment pointing at it.
- Docstrings: `:358` and `:691` — "Inclusive supreme upper bound" gains "(the run's data edge — no
  candle past it is read by the probe or by the MS it drives; Plan A)".

### 3.4 `structure/structure_engine.py`
`compute_bounded_structure` docstring (`end_idx` parameter): "Inclusive upper bound … The run never
writes/emits past it" → "Inclusive upper bound. The run is identical to an unbounded run on the frame
truncated at `end_idx` (nothing past it is read — MarketStructure §2 of Plan A); no event carries an
`idx` past it (asserted)."

### 3.5 Explicitly unchanged
`entity_df_mutation.py`, `pooled_structure_build.py` (they slice to the bound; no edit), `range_label.py`
(labels stay full-frame; the MS rule makes them bound-safe), `pattern_engine.py`, every caller of
`MarketStructure(end_idx=…)` (they inherit the fix). `range_max_k` stays 5 and stays shared with the
detector's max confirmation offset (`idx+5`) and `RangeLabelConfig.max_lookahead` (5) — the three are
one horizon; do not decouple them here.

---

## 4. Audit — every forward read in the bounded-run paths, classified

"Forward read" = any read of a candle index `> i` while processing anchor/candle `i`. Classification:
**FIX** (changed by this plan), **BOUNDED** (already limited to `<= effective_end` or to an explicit
`[lo, hi]` inside it), **EQUIVALENT** (reads past the bound but the result is provably what the
truncated frame gives), **WRITE** (writes, not reads, past the bound), **OUT** (a finding, not in
scope — §8).

| Site | Read | Class | Note |
|---|---|---|---|
| `market_structure.py:897, :1535` `D` | pattern applies + back-fill to `i+5` | **FIX** L1 | |
| `:1092` `is_range_confirm_idx` (label read) | confirming close to `i+5` | **FIX** L2 | label computed full-frame by `range_label.py` |
| `:710` watch expiry, `:784` rewind target, `:482` rewind clamp | `i+5` / `n-1` | **FIX** L4 | §2 judgement call |
| `structure_patterns.py` six guards | pattern candles / confirmations to `idx+5` | **FIX** L3 | via `n_visible` |
| `:925 for k in range(i, apply_idx)` and `:952/:1549 for k in range(i, min_d)` back-fills | | BOUNDED after L1/L2 (`apply <= D`, `min_d <= D`) | |
| `:941 next_i = apply_idx + 1` | loop control | BOUNDED (`while i <= effective_end`) | |
| `_get_cts0_tfb` → `find_true_first_breakout(bp, start, effective_end, …)` | | BOUNDED (`hi = min(upper, n-1)`, `est > hi` dropped, all four detectors enumerated — no priority pre-emption) | after §3.2 also bounded by `bp` |
| `_maybe_expire_reversal_watch` `self._l[anchor]` / `self._h[anchor]` | backward | BOUNDED | |
| `_select_bos_on_breakout(apply_idx)`, `_initial_bos_before_first_cts(cts_idx)`, `_cts_from_breakout_event` (span `[start, confirmation]`, `confirmation <= apply <= D`) | backward / `<= D` | BOUNDED | |
| `_replay_step_no_patterns(i)` → `_update_active_range(i)`, `_maybe_update_cts_pre_confirm(i)`, `_maybe_confirm_cts_via_proximity(i)` | candle `i` + resolver-derived zones from events so far | BOUNDED | resolver protocol (Part 4 §13.5.b) derives from emitted events |
| `_update_cycle0_data` → `has_unfilled_imbalance(df, lo, hi, check_to_idx=cts_idx)` | as-of `cts_idx` | EQUIVALENT for fills (as-of) — **OUT** for instance existence at the bound (an FVG whose `c2 == B` exists on the full frame because `c3 = B+1` was seen; absent on the truncated frame). `has_unfilled` only feeds the in-flight Scenario-2 / POI-inner snapshot. | see §8; the §5.1 test will show it if it ever matters on the fixtures |
| `_schedule_reversal_from_anchor` → `detect_best_for_anchor` + "apply within the watch window" | to `idx+5` | **FIX** L3 (detector) + BOUNDED (watch window `<= effective_end` after L4) | |
| `run()` terminal stamping `self.df.loc[first_rev_idx:, "market_state"] = "reversal"` | rows `> effective_end` | WRITE | forward stamp for chart/invariant consistency; every live caller's frame ends at the bound or discards the df. Unchanged. |
| `_rewind_to` replays from `i = 0`, not `start_idx`; the rebuild ignores earlier jump requests | | OUT | pre-existing rewind semantics, identical in bounded and truncated runs. §8 |
| `unified_probe._run_phase1`: `find_true_first_breakout(bp, cur, upper=end_idx)`, `_select_extreme_retrace_candidate(df, est+1, end_idx)`, `_evaluate_reset_conditions(df, candidate)`, `_bos0_inner_at_start(df, start)` | `[lo, hi]` explicit / single candle / backward | BOUNDED (EQUIVALENT for the detector reads inside `_anchor_candidates` — all four detectors enumerated, `est > hi` dropped) | after §3.3 bounded by `bp` too |
| `unified_probe._run_phase2`: `_make_market_structure(df_probe, end_idx=end_idx)` then reads `probe_events`, `df_probe["market_state"]` rows | | **FIX** via MS | the observed leaks |
| `structure_engine` main reversal probe: `unified_probe(df2.copy(), end_idx=reversal_apply_idx, enable_phase2=False)` | Phase 1 only | EQUIVALENT (above) | **H1 must stay byte-identical — the tripwire for this claim** |

---

## 5. Tests

### 5.1 New — the property test (`tests/test_ms_bounded_equals_truncated.py`)
The §2 definition, checked for **every** bound, so it cannot pass by fixture luck:
```python
raw = <ohlc rows>                                  # _make_reversing_data(), _make_uptrend_data(), + §5.2 fixtures
full = _prepare_df(raw)                            # tests/test_bounded_structure._prepare_df: classification
                                                   #   + detect_patterns (pat*, is_range*) + compute_imbalance
for B in range(start + 3, len(raw) - 1):
    bounded   = compute_bounded_structure(full, start, sd, end_idx=B)
    truncated = compute_bounded_structure(_prepare_df(raw[:B + 1]), start, sd, end_idx=None)
    assert ev_sig(bounded.events) == ev_sig(truncated.events)
    assert rows_equal(bounded.df.iloc[start:B + 1], truncated.df.iloc[start:B + 1], STRUCTURE_OUTPUT_COLS)
    assert bounded.reversal_idx == truncated.reversal_idx
```
`ev_sig` = sorted `(type, idx, round(price, 6), confirmed_at, cycle_id, reason/via if present)`; compare
`meta` minus nothing volatile — if a meta field legitimately differs (it should not), name it in the test.
`STRUCTURE_OUTPUT_COLS` = the MS output columns (`market_state`, `cts_*`, `bos_*`, `range_*`,
`structure_id`, `cycle_id`, …; take the list from `_ensure_output_cols`). Parametrise over `sd = ±1` and
over the fixtures. Runtime: MS on ~200-300 synthetic candles is milliseconds; a few hundred bounds ×
fixtures stays well under a few seconds. **Run this test at the base first** — it must FAIL there on at
least one bound (that failure is the defect; record which bound and which of L1–L4 explains it in the
test's docstring). Note `_prepare_df` recomputes `compute_imbalance` on the truncated rows too, so a
difference caused by instance existence at the bound (§4, OUT) would surface here — if it does, record
it (§8) and decide with the user whether it joins Plan A; do not silently exclude it.

### 5.2 New — boundary fixtures (each also fed to §5.1)
Synthetic OHLC built with the `_seg`-style helpers in `test_bounded_structure.py`; each fixture asserts
the specific mechanism so the failure message names it:
- **L2 range at the bound:** a range candle `i` whose first confirming close is at `i+3`. With `B = i+2`
  → no `RANGE_STARTED`; with `B = i+3` → `RANGE_STARTED.idx == i+3`.
- **L1 pattern past the bound:** a breakout pattern anchored at `i` confirming at `i+2`. `B = i+1` → no
  `CTS_ESTABLISHED` for it; `B = i+2` → established.
- **L3 priority pre-emption:** an anchor `i` where a 2-candle pattern SUCCEEDs at `i+1` and a
  `continuous` SUCCEEDs at `i+2`. `B = i+1` → the 2-candle pattern establishes CTS at `i+1` (today:
  nothing). `B >= i+2` → `continuous` wins (unchanged).
- **L4 watch at the bound:** a close-break of `bos_threshold` at `i` with no reversal pattern in
  `[i, i+5]`. `B = i+2` → a `BOS_THRESHOLD_UPDATED(reason="probe_no_break")` at `B` and no reversal
  (the data-edge behaviour); `B >= i+5` → the same at `i+5` (unchanged).
- **Post-run assert:** any of the above at the base would make §3.1.6 fire — it is a guard, keep one
  negative test that monkeypatches `_effective_end` past the frame and expects the assert.

### 5.3 Existing tests
| test | action |
|---|---|
| `test_pooled_structure_build::test_capped_run_emits_nothing_past_cap` | strengthen: also `ev.idx <= cap` (it becomes a real guarantee, not fixture luck) |
| `test_pooled_structure_build::test_events_knowable_at_are_causal_in_end_idx` | keep as is. It asserts prefix-equivalence on a fixture whose tail is quiet; it stays true there. If it fails after Plan A, that is the documented §2 tail divergence on that fixture — then restrict it to events with knowable-at `<= cap - range_max_k` and say why in its docstring. Plan C repoints it to the slicing builder anyway. |
| `test_bounded_structure::test_end_idx_caps_before_reversal`, `::test_first_segment_events_match` | keep; expected green |
| `test_unified_probe` | add: on its Phase-2 fixture, every event of the bounded MS run has `idx <= end_idx` (redundant with the assert, but documents the contract at the probe level) |
| `test_sub_chain.py`, `test_first_trigger_migration.py`, `test_sub_structure_pool.py` | no change expected (sub builds slice to the bound). Any change there = the slice claim in §1 is wrong — investigate before proceeding |

---

## 6. Acceptance (`/compare` against the P1 revert save + chart review)

1. **H1: all CSVs byte-identical.** Main runs are `end_idx=None` (every edit is `n-1 → effective_end`
   and they are equal there); the main reversal probe is Phase-1 only and equivalent (§4). Any H1
   delta = the equivalence claim is wrong → **stop and investigate** before looking at M15.
2. **M15: every delta must trace to a first_confluence probe Phase-2 outcome.** Run
   `debug/probe_fc_finalize.py` at the base and after the fix; diff the five rows. Predicted:

   | cycle | today | after Plan A | why |
   |---|---|---|---|
   | (0,0) | start 454, finalize 1020, `second_cts_reached` | unchanged | leaked candles 1722–1725 lie outside the retrace window `[459..783]`; the 2nd CTS_EST at 1020 is far from the bound 1721 |
   | (0,1) | 2365 / 2608 / `second_cts_reached` | unchanged | no leak observed; bound 2609 |
   | (1,0) | 2803 / **2844** / `no_retrace` | finalize **2843** (`no_retrace`, else-branch: cycle 0 no longer confirms inside the window) ; `starting_idx` **2803 may move** — the retrace window widens to `[2761..2843]` and a new most-extreme candidate at 2842/2843 could pass the reset | L1 |
   | (1,1) | 2915 / 3047 / `no_retrace` (else) | unchanged finalize (already the else-branch `end_idx`); events past 3047 gone | L2 |
   | (1,2) | 3304 / 3621 / `no_retrace` (else) | unchanged | no leak observed |

   A change in any row other than (1,0)'s finalize is possible only through L3/L4 at that cycle's bound
   (a pre-emption or an open watch at `probe_end_idx`). If one appears, it is **allowed but must be
   named**: show the anchor/watch at the bound in the Phase-2 run (`ms.debug = True` prints
   `[POST_STEP]`/`[RV_EXPIRE]`) and record it in the ground-truth memory table.
3. **Sub builds are byte-identical** except the sids whose probe row changed: under 3.2a the FC(1,0)
   sid (`2803/−1`, `inactive` at 3611 in the baseline) carries the finalize as `start_trigger_idx`
   (2844→2843 in `_sids.csv` / sid meta) and, if `starting_idx` moved, a different structure. Its KL /
   POI / fib / WVMI rows follow. Everything else in every M15 CSV: identical.
4. Log grep (`warning|skipping|unavailable|degenerate|pending|no sid`): no new lines.
5. Display the `=== Replay Timing ===` block. Expected: unchanged (the fix removes work).
6. **Pause for chart review** (the visible change is at most one `inactive` sub's outline), then
   `/commit-save`.
7. Anything else = regression.

---

## 7. Docs in the same commit
- **`LANDMINES.md`** "MarketStructure Range Look-Ahead Leaks Past `end_idx`": retitle "Bounded MS Runs
  Must Not Read Past `end_idx` (FIXED — Plan A, commit …)"; rewrite the body to the §2 definition, the
  four sites L1–L4, the "clamp `D` alone is not enough — `RANGE_STARTED` is stamped at the label's
  `confirm_idx`" correction, the post-run assert as the guard, and the §2 "not prefix-equivalence"
  paragraph (so nobody writes a clip-equivalence test and calls its failure a bug). Cross-link "Probe
  `end_idx` Is the Supreme Bound".
- **`structure/MARKET_STRUCTURE_SPEC.md`**: add a short "Bounded runs (`end_idx`)" subsection stating
  the definition, the shared 5-candle horizon (`range_max_k` = detector confirmation offset =
  `RangeLabelConfig.max_lookahead`), and the assert.
- **`PART4_REFACTOR_SPEC.md`** §5 (`:~823` "all 3 once the MS bounds leak is fixed") → state the measured
  post-fix table; §17.11's Plan A line → "landed, commit …".
- **`GOTCHAS.md`** `:~1601-1606` (finalize native-vs-mapped entry) — still correct; add "fixed by Plan A"
  to its bounds-leak sentence.
- **Memory:** `project_sub_structure_pool_architecture.md` "PRE-EXISTING BUGS" → Plan A FIXED (commit);
  `reference_pool_redesign_groundtruth.md` FC table row (1,0) → measured values, drop the `*` footnote;
  `MEMORY.md` constraint line "currently VIOLATED by the range look-ahead (Plan A)" → fixed; Plan C §0 P1
  wording if `starting_idx` 2803 moved.
- `debug/probe_fc_finalize.py`: keep; add a printed `max_event_idx` per cycle next to `m15_end` so the
  leak measure is one column, not an instrumented run.

---

## 8. Findings outside scope (record, do not fix here)
- **`_rewind_to` replays from candle 0, not `start_idx`** (`market_structure.py:492`), and the rebuild
  **ignores earlier jump requests** (`:497-500`) — a rebuild can differ from the first pass wherever an
  earlier watch expiry had rewound. Deterministic and identical between bounded and truncated runs, so
  not a Plan A concern; on the reference window H1 has 0 expiries, M15 confluence 15, counter 3. Worth
  its own look (does the rebuild's `_replay_step_no_patterns(k)` for `k < start_idx` clobber a prior
  sid's rows on the main path? seeded arrays make the flush safe, the rebuild writes are the question).
- **Imbalance instance existence at the bound** (§4): an FVG with `c2 == B` exists on the full frame only.
  `has_unfilled_imbalance` is as-of for *fills*, not for *existence*. Feeds only the in-flight
  Scenario-2 / POI-inner snapshot. Same family as `feedback_in_flight_vs_downstream_resolver`.
- **Prefix (clip) equivalence does not hold in the last 5 candles** (§2) — inherent; Plan C's model
  (geometry to the data edge, windows clipped from one run) is the answer, and `knowable_at_idx`'s
  half-clip of `CTS_ESTABLISHED` / `REVERSAL_CANDIDATE` remains a separate, known gap.
- Terminal reversal stamping writes rows past the bound (WRITE, harmless for every live caller).

## 9. Definition of done
§5 tests green (the property test fails at the base on at least one bound and passes after); §6.1–6.4
satisfied and every M15 delta named by mechanism; chart review; §7 docs in the same commit;
`/commit-save`; memory status updated; then Plan B.
