# Plan A — Bounded MarketStructure Runs Read Nothing Past Their Bound

**Status:** READY TO IMPLEMENT (written 2026-09-19; cold-reviewed the same day by two fresh agents —
code-reference audit + implement-on-paper — and a decision-coverage pass; every finding applied; the
L5 scope resolution the review added was confirmed by the user the same day). **No open items.**
**Defect record:** `LANDMINES.md` "MarketStructure Range Look-Ahead Leaks Past `end_idx`" (the plan
corrects it — the original entry named only site L1 of the five below). §7 retitles that entry; when it
does, update this header's citation in the same commit.
**Ground truth:** `memory/reference_pool_redesign_groundtruth.md` (the first_confluence probe table;
"Observed leaks past `end_idx`"). Reproduce with `engine_v2/debug/probe_fc_finalize.py` (OANDA, ~2 min).
**Sequencing:** first of three — `git revert 9fd3143` → **Plan A** → Plan B → Plan C (PART4 §17.11).

This document is the implementation contract. A session with no memory of the design discussion can
execute it. Where it says "decided", do not re-open. Line numbers are at `e6292a3` (HEAD before the
revert); `9fd3143` touches only render code (`entity_df_mutation.py`, `orchestrator.py`), so every
reference below survives the revert.

---

## 0. Prerequisites, base, sequencing

| Step | Must be true before starting |
|---|---|
| P0 | `9fd3143` (Stage 3.2b) has been **`git revert`ed** on `week8-volmom-multitf` (kept in history). Tree = Stage 3.2a semantics. |
| P1 | The revert has its own **`/commit-save`** (78 s) and its `/compare` against `20260708_015339_c932610` is **byte-identical on every CSV** (the revert restores 3.2a exactly; if not, stop — the revert is wrong). This save is Plan A's `/compare` baseline. |
| P2 | 418 tests green at the base. |

Plan A is **one behavioural change** ("a bounded MS run reads only candles ≤ its bound") ⇒ one replay,
one `/compare`, one chart-review pause, one `/commit-save`. Do not bundle Plan B into it.

---

## 1. The defect, precisely

A bounded run is `MarketStructure(end_idx=B)` (via `compute_bounded_structure(end_idx=B)`). Its main loop
is bounded (`effective_end = min(n-1, B)`, `while i <= effective_end`, `market_structure.py:414-422`), but
five things inside the loop still see the dataframe end `n-1`, not `B`:

| # | Site | What leaks | Observed on the reference window |
|---|---|---|---|
| L1 | `_step_anchor` `:897` and `_post_apply_range_check` `:1535`: `D = min(i + self.range_max_k, n - 1)` | `D` is the horizon for **pattern applies** (`apply <= D`, `:1015/1040/1060`) and for the range **back-fill** (`for k in range(i, min_d)`, `:952/1549`). A pattern anchored at `i <= B` applies at `apply <= i+5`, possibly `> B` — `_apply_pattern_at_apply_idx` fires there (a `continuous` SUCCESS at `i+2 > B` establishes a CTS past the bound), and the back-fill steps `_replay_step_no_patterns(k)` run for `k > B` (per-candle CTS updates, proximity confirmation, BOS probes). | FC(1,0): `CTS_CONFIRMED@2844` on a run bounded at 2843 — the **only** reason its `finalize_idx` is 2844 and its retrace window `[2761..2841]` instead of `[2761..2843]`. FC(0,0): events to 1725 on a bound of 1721. Property check at HEAD: the reversing test fixture diverges from its truncated twin on 22 of 57 bounds, first at `B=3` (L1). |
| L2 | `_finalize_range_candidate_offline` `:1127-1144`: `RANGE_STARTED` and `STATE_CHANGED→RANGE` are stamped at **`confirm_idx`** = the pre-computed `is_range_confirm_idx` label (`range_label.py:62-77`: the first close in `[i+2, i+5]` inside candle `i`'s range), read raw from the df (`:1092`, re-read `:1110`) and **never compared to the bound**. | A range candidate at `i <= B` whose confirming close is at `confirm_idx > B` is activated and its events stamped in the future. **Clamping `D` does not touch this path** — the LANDMINES entry's "clamp `D`" is necessary but not sufficient. | FC(1,1): `RANGE_STARTED` / `STATE_CHANGED@3050` on a run bounded at 3047. |
| L3 | `BreakoutPatterns` guards on `len(self.df)` (`structure_patterns.py:115/156/181/349/439/529`) | The detector reads candles past `B` to decide SUCCESS / CONFIRMED. **Once L1 is fixed** MS drops a candidate whose apply is `> D` — usually equivalent to "not knowable yet", but `detect_best_for_anchor` returns ONE pattern per anchor by **priority** (`:672-683`: `continuous` SUCCESS at `idx+2` pre-empts a 2-candle SUCCESS at `idx+1`). At `B = idx+1`: full frame → `continuous` wins and is dropped, **no pattern**; a frame that ends at `B` → `continuous` is `None`, the 2-candle pattern is returned and applied. A future candle changed the decision. (At HEAD, before L1, the `continuous` is simply *applied* past the bound — that is L1.) | Real by code reading; needs a 3-candle and a 2-candle pattern on one anchor exactly at a bound (§5.2 fixture). |
| L4 | `_start_reversal_watch` `:710`: `reversal_watch_expires_idx = min(i + range_max_k, len(self.df) - 1)`; `_maybe_expire_reversal_watch` `:784`: `jump_to = min(anchor + 1, len(self.df) - 1)`; `_rewind_to` `:481-482` clamps to `n-1`. | **How the watch actually works (verified):** a watch survives its anchor only if `_schedule_reversal_from_anchor` (`:816-845`) found a reversal pattern with `apply_r <= expires_idx`; otherwise it is cleared at once (`rv_anchor_failed`, `:640-650`). Expiry (`probe_no_break` + rewind) fires only when the pending apply equals `expires_idx`, because `_maybe_expire_reversal_watch` runs **before** `_maybe_apply_pending_reversal` in the per-candle step (`:578-581`). So with `expires_idx` clamped to `n-1` instead of `B`: a reversal pattern applying in `(B, i+5]` is *scheduled* (the run then ends at `B` with a watch and pending reversal open), whereas a frame ending at `B` never schedules it. | Not measured; the §5.2 L4 fixture pins it. |
| L5 | The two **resolvers** MS calls with `self.df` — `_bos_inner_resolver` at `BOS_CONFIRMED` (`:1468-1472` → `compute_bos_inner_from_event`, `kl_zones_v1.py:1102-1145`) and `_poi_inners_resolver` at `CTS_ESTABLISHED` / `CTS_UPDATED` (`:1912-1922`) — derive a base pattern with `identify_base_pattern`, whose inside-bar scan reads `[anchor-5, min(len(df)-1, anchor+5)]` (`kl_zones_v1.py:58-59`), whose 2-candle / star paths read `anchor+1` (`:639-643`, `:662-673`) and whose `zone_thresholds` reads `base_idx+1/+2` (`:540-553`). Clamped to the **frame**, not the bound. | A BOS extreme within 5 candles of `B` gets its inner from candles `> B`; that inner feeds `_maybe_confirm_cts_via_proximity` at candles `<= B` → a `CTS_CONFIRMED` that a frame ending at `B` would not (or would differently) produce. **Found by the cold audit; my original table classified these reads as "bounded".** | Not measured; on the two test fixtures the read past the current candle occurs (bos_idx=0, confirmed_at=2, reads to 5) but the inner happened to be equal. |

**Decided (user-confirmed 2026-09-19): L5 is IN scope (§3.1.7), via a truncated *view* of the frame
handed to the resolvers, not via signature changes in `zones/`.** Without it the §2 definition is false and the
§5.1 property test can only pass by fixture luck (any `BOS_CONFIRMED` has bounds within 5 candles after
it). The zone-derivation reads are the same class as L1–L4 (clamped to `len(df)-1`), the fix is
two call sites inside MS, and the main path is untouched by construction (`end_idx=None` → no view). Its
cousin **L5b** — the *probe's* ad-hoc BOS_0 derivation at a reset candidate (`unified_probe.py:342-353`
`_bos0_inner_at_start` → `build_ad_hoc_bos0_reference_zone` → the same `identify_base_pattern`, reading up
to `candidate+5` past `probe_end_idx`) — is deliberately **OUT** (§8): the probe's bound is a *search*
bound (those candles exist at trigger time, so it is a reproducibility gap, not a causality one), and
touching it can move the H1 sid-1 start through the main reversal probe, breaking the H1 byte-identity
that is this plan's regression tripwire. It gets its own decision later.

**Why it matters (unchanged from LANDMINES):** a bounded run is not reproducible from its stated bound;
`unified_probe`'s docstring ("Inclusive supreme upper bound", `unified_probe.py:691`) and
`compute_bounded_structure`'s ("The run never writes/emits past it", `structure_engine.py:871-872`) are
false today; the probe's `finalize_idx` — and through the retrace window, `starting_idx` = the pool
key — can depend on candles the probe was told not to use.

**Who is actually affected (verified twice).** Every sub geometry build slices its frame to the bound
first — `_build_or_get_sub_geometry` takes `entity_df.iloc[slice_begin:run_cap_abs + 1]`
(`entity_df_mutation.py:1076`), passes `end_idx = run_cap_in_slice` (`:1102-1110`) and re-derives the
`is_range_*` labels on the slice (`:1088-1093`); `build_one_sid` builds through it (`:1219`) — so there
`n-1 == effective_end`, the resolvers already see a frame ending at the bound, and **Plan A is a no-op by
construction**. Main runs pass `end_idx=None` (`structure_engine.py:208-213`). The only production paths
that run MS on a frame longer than the bound are:
- `unified_probe._run_phase2` (`df_probe = df.copy()` of the whole M15 entity frame, `unified_probe.py:491`,
  `end_idx = probe_end_idx`, `:502-510`, scan mode `enforce_cts0_new_extreme=True`) — the
  first_confluence probe. **This is where every observed leak came from.**
- `unified_probe._run_phase1` / `find_true_first_breakout` (all five trigger types + main reversals) —
  L3 only, and provably equivalent already (§4), so no output change is expected there.
- `pooled_structure_build.build_structure_geometry` (`:67-75`, passes the whole df with `end_idx=run_cap`) —
  test-only twin, deleted by Plan C.
- `compute_structure_scenario_3` / `compute_structure_from_start` (`structure_engine.py:420/553/757/705`) —
  reachable only through the empty `_LEGACY_PROBE_USE_CASES` escape hatch and `test_scenario3.py`.

So the expected replay footprint is small (§6). The fix is still made **inside MS and the detector**,
not at the callers, so that every present and future bounded run has the same semantics.

---

## 2. Semantics (decided)

**Definition.** A bounded run `MarketStructure(df, start_idx=s, end_idx=B).run()` produces exactly the
events and output rows of `MarketStructure(df_B, start_idx=s, end_idx=None).run()` where `df_B` is the same
frame **truncated to `[0, B]`** (with its look-ahead feature labels recomputed on the truncated frame).
`effective_end` **is the run's data edge**: nothing past it is read, and everything the run does at
the edge is what it already does at the real data edge today.

**Known residual (documented, not fixed here):** `df.attrs["imbalances"]` is a full-frame instance list.
An FVG instance whose `c2 == B` exists only because `c3 = B+1` was seen, and a merged run ending at `B-1`
gets its bounds from `l/h[B+1]` (`imbalance.py:55-75, 92-96`); `has_unfilled_imbalance` is as-of for
*fills* (`types.py:213-222`), not for existence/bounds. Consumers: `_update_cycle0_data` (`:1957-1961`)
and `_refresh_poi_inners_for_cycle` → `find_ic_candidates` (`poi_zones.py:225-238`) — the in-flight
Scenario-2 / POI-inner snapshot only. Observable path: a CTS raw-update at `B-1` → different
`poi_inners_for_cycle` → `_maybe_confirm_cts_via_proximity(B)` differs. Rare; §5.1 will expose it if a
fixture has an FVG straddling `B`, and §8 says what to do then.

**What this is NOT.** It is not "clip of the natural-end run ≡ bounded run" (prefix-equivalence). The
two can differ in the last `range_max_k` (5) candles before `B`, inherently: the natural-end run may be
back-filling a pattern that applies past `B` (rows written with `freeze_range=True`, next anchor
`apply+1`), while the bounded run processes those candles as anchors (unfrozen range expansion,
range candidates, a pattern the natural run's priority rule pre-empted); a reversal pattern pending
past `B` is scheduled in one run and not the other; a **pending reversal whose apply is exactly `B` is
discarded as a false break** in the bounded run (expiry runs before the pending apply, `:578-581`) while
the natural run reverses there; and an extra rewind rebuilds from 0 ignoring earlier jump requests
(`_rewind_to`, `:489-503`), which can change events before the rewind point. This is the 5-candle
pending-confirmation nature of MS, not a bug, and it is exactly why the pool (Plan C) runs geometry to
the **data edge** and clips *windows* out of one run rather than relying on bounded builds. Do not try
to make prefix-equivalence hold; do not "look ahead but suppress emission" (rejected in LANDMINES —
it keeps future information inside the state machine).

**L4 — decided (user-confirmed 2026-09-19): option (a), truncation semantics for the watch too.**
`expires_idx = min(i + range_max_k, effective_end)`. Consequences, all identical to today's data edge:
a reversal pattern whose apply is `> effective_end` is never scheduled (anchor fails at `i`); one whose
apply is `< effective_end` reverses as today; one whose apply is **exactly** `effective_end` is discarded
as a false break (`probe_no_break` at `effective_end`, rewind to `anchor+1`) because expiry precedes the
pending apply in the per-candle step. The rejected alternative (leave `expires_idx` at `n-1`) would keep
a watch and a pending reversal open past the bound, making bounded ≠ truncated whenever a watch
straddles the bound.

---

## 3. Changes (file-level)

Edit order that keeps every intermediate state runnable: §3.2 (detector) → §3.1.1–2 → L1/L2/L4/L5 →
§3.3/§3.4 docstrings → the post-run assert **last** (it would fire at every intermediate step).

### 3.1 `structure/market_structure.py`
1. **`__init__`** (after `self.end_idx = …`, `:324`, and before `self._bp = BreakoutPatterns(self.df)`, `:364`):
   ```python
   n = len(self.df)
   self._effective_end = n - 1 if self.end_idx is None else min(n - 1, int(self.end_idx))
   ```
   `run()` (`:413-416`) and `_get_cts0_tfb` (`:1718-1721`) use `self._effective_end` instead of
   recomputing it. `run()` keeps its own `n` (`:407` `i >= n`, `:420` `_init_output_arrays(n)`).
2. **`BreakoutPatterns(self.df, end_idx=self._effective_end)`** (`:364`).
3. **L1:** `:897` and `:1535` → `D = min(i + self.range_max_k, self._effective_end)`. The `n = len(self.df)`
   locals (`:896`, `:1534`) become unused — delete them.
4. **L2:** `_is_range_candle_given_confirm` (`:1092-1094`):
   ```python
   confirm_idx = int(self.df.iloc[i].get("is_range_confirm_idx", -1))
   if confirm_idx < 0 or confirm_idx > self._effective_end:
       return (False, None)          # the confirming close has not happened yet
   ```
   This is the whole L2 fix: both call sites (`:948`, `:1539`) go through it, and
   `_finalize_range_candidate_offline` re-reads the label (`:1110`) only after `range_confirmed` was True.
   Equivalent to recomputing `apply_is_range_labels` on the truncated frame (the label locks the FIRST
   confirming close, `range_label.py:61-74`; if that close is `> B` no close `<= B` confirmed it, and
   `is_range == 0` is already rejected at `:1088`).
5. **L4 (per §2):** `:710` → `min(int(i) + int(self.range_max_k), self._effective_end)`; `:784` →
   `min(anchor + 1, self._effective_end)`; `_rewind_to` `:481-482` → `if jump_to > self._effective_end:
   jump_to = self._effective_end` (dead once `:784` is changed — keep for symmetry; the local `n` at
   `:476` becomes unused, delete it).
6. **Post-run assert**, placed **right after the `while` loop** (`:447`, before `_flush_output_arrays`,
   so it fires independently of df state and before the invariant checks). Unconditional — O(events),
   and it is the invariant this plan exists for:
   ```python
   if self.events:
       _max_idx = max(int(ev.idx) for ev in self.events)
       assert _max_idx <= self._effective_end, (
           f"[market_structure] event past effective_end: max idx {_max_idx} > {self._effective_end}")
   ```
   Every emit site stamps `i`, the anchor, an extreme inside a pattern span (≤ apply), or `confirm_idx`
   (≤ `effective_end` after L2); apply ≤ `D` ≤ `effective_end` after L1 (scan-mode applies are
   `tfb.est_idx <= hi`); the rewind rebuild reuses `_step_anchor` with the same `D`. So after L1+L2+L4 the
   assert cannot fire (both reviewers enumerated all 14 emit sites). If it fires, a forward read was
   missed — do not weaken it.
7. **L5 — resolvers see the truncated frame.** One helper:
   ```python
   def _resolver_df(self) -> pd.DataFrame:
       """The frame a resolver may read: truncated at the run's data edge. Main runs
       (end_idx=None) get self.df itself — no slicing, no behaviour or perf change."""
       if self._effective_end >= len(self.df) - 1:
           return self.df
       return self.df.iloc[: self._effective_end + 1]
   ```
   and `self.df` → `self._resolver_df()` at the two call sites `:1468-1472` (`_bos_inner_resolver`) and
   `:1912-1922` (`_poi_inners_resolver`). `iloc` on a RangeIndex frame keeps `loc == iloc` and propagates
   `attrs` (the resolvers use `df.loc`, `df.index` membership, `len(df)` and `df.attrs["imbalances"]` —
   the last is the §2 residual). Per-call cost is one pandas slice on the probe frames only (sub
   builds and main hit the fast path); measure in the timing block, expect no visible change.
8. **Delete the stray debug print** `:1025-1036` (`if i in (102, 103): … self.df.iloc[i+1] …`) — a
   leftover that prints to stdout and reads `i+1` (guarded only by `omc is not None`; `n_visible` makes
   it safe but it has no business in production). Behaviour-preserving.
9. **`run()` docstring**: replace "If end_idx is set, processing stops after that idx (inclusive)" with
   the §2 definition (two sentences: truncation semantics; not prefix-equivalence).

Nothing else in MS changes. The main-path behaviour (`end_idx=None`) is untouched by construction:
every edit is `n-1 → self._effective_end` (equal there) or a fast-path `self.df`.

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
(`:107-108`) reads `i+1` but its only callers (`:386/397/411/422`) sit behind the `:349` guard. `_row`
(`:58`) and the `_cols` arrays stay full-length — safety rests solely on the six guards; keep them
together. Add one docstring line to `detect_best_for_anchor` under its "Notes": "With `end_idx`, candles
past it do not exist for this detector: a SUCCESS/CONFIRMED that would need them is reported as `None` /
unconfirmed, never as a candidate to be dropped later."
`pattern_engine.py:49` (offline `pat*` marker columns over the whole frame) keeps `BreakoutPatterns(df2)`
unbounded — MS never reads the `pat*` columns (verified: its df reads are o/h/l/c `:318-321`,
`market_state` `:459`, `is_range` `:1088`, `is_range_confirm_idx` `:1092/:1110`, `candle_type` `:1096`,
`direction` `:1097`, `time` `:2560` and its own output columns; the detector's `_ROW_FIELDS` `:17-24`).

### 3.3 `structure/unified_probe.py`
- `:715` → `bp = BreakoutPatterns(df, end_idx=end_idx)`. Phase 1 is then bounded by construction rather
  than by the `est > hi` drop in `find_true_first_breakout` (§4 — equivalent; expect no change).
  `find_true_first_breakout` keeps `hi = min(upper_idx, n-1)` (`true_first_breakout.py:224`).
- Phase 2: the MS assert (§3.1.6) already covers its run; add nothing but a comment pointing at it.
- Docstrings `:358` and `:691` — "Inclusive supreme upper bound" gains "for the breakout search, the
  retrace window and the MS run Phase 2 drives (no candle past it is read there — Plan A). The ad-hoc
  BOS_0 zone derivation at a reset candidate still reads up to 5 candles past it (Plan A §8, L5b)."

### 3.4 `structure/structure_engine.py`
`compute_bounded_structure` docstring (`:871-872`): "Inclusive upper bound … The run never writes/emits
past it" → "Inclusive upper bound. The run is identical to an unbounded run on the frame truncated at
`end_idx` (nothing past it is read — Plan A §2, with the `attrs["imbalances"]` residual noted there); no
event carries an `idx` past it (asserted)."

### 3.5 Explicitly unchanged
`entity_df_mutation.py` (slices to the bound; no edit), `pooled_structure_build.py` (does **not** slice —
passes the whole df with `end_idx=run_cap` — but is test-only and deleted by Plan C; no edit),
`range_label.py` (labels stay full-frame; the MS rule makes them bound-safe), `pattern_engine.py`,
`zones/*` (L5 uses a view, no signature change), every caller of `MarketStructure(end_idx=…)` (they
inherit the fix). `range_max_k` stays 5 and stays shared with the detector's max confirmation offset
(`idx+5`), `RangeLabelConfig.max_lookahead` (5) and the inside-bar scan half-width (5) — one horizon; do
not decouple them here.

---

## 4. Audit — every forward read in the bounded-run paths, classified

"Forward read" = any read of a candle index `> i` while processing anchor/candle `i`. Classification:
**FIX** (changed by this plan), **BOUNDED** (already limited to `<= effective_end` or to an explicit
`[lo, hi]` inside it), **EQUIVALENT** (reads past the bound but the result is provably what the
truncated frame gives), **WRITE** (writes, not reads, past the bound), **OUT** (a finding, not in
scope — §8). Verified line by line by the cold audit.

| Site | Read | Class | Note |
|---|---|---|---|
| `market_structure.py:897, :1535` `D` | pattern applies + back-fill to `i+5` | **FIX** L1 | |
| `:1092` `is_range_confirm_idx` (label read) | confirming close to `i+5` | **FIX** L2 | label computed full-frame by `range_label.py` |
| `:710` watch expiry, `:784` rewind target, `:482` rewind clamp | `i+5` / `n-1` | **FIX** L4 | §2 |
| `structure_patterns.py` six guards | pattern candles / confirmations to `idx+5` | **FIX** L3 | via `n_visible` |
| `:1468` `_bos_inner_resolver(self.df, bos_idx, …)` → `identify_base_pattern` inside-bar scan to `bos_idx+5`, 2-candle/star `+1`, thresholds `+1/+2` | to `bos_idx+5`, clamped to the frame | **FIX** L5 | consumed by proximity confirmation at candles ≤ `B` when `bos apply < B` |
| `:1912` `_poi_inners_resolver(self.df, …)` → same derivation on the BOS/CTS anchors + `attrs["imbalances"]` | to `anchor+5` (candles); full-frame instances (attrs) | **FIX** L5 for the candle reads; **OUT** for the attrs residual (§2) | |
| `:925 for k in range(i, apply_idx)` and `:952/:1549 for k in range(i, min_d)` back-fills | | BOUNDED after L1/L2 (`apply <= D`, `min_d <= D`) | |
| `:941 next_i = apply_idx + 1` | loop control | BOUNDED (`while i <= effective_end`) | |
| `_get_cts0_tfb` → `find_true_first_breakout(bp, start, effective_end, …)` | | BOUNDED (`hi = min(upper, n-1)` `:224`, `est > hi` dropped `:245`, all four detectors enumerated `:166-171` — no priority pre-emption) | after §3.2 also bounded by `bp` |
| `_maybe_expire_reversal_watch` `self._l[anchor]` / `self._h[anchor]` | backward | BOUNDED | |
| `_select_bos_on_breakout(apply_idx)` `:2127`, `_initial_bos_before_first_cts(cts_idx)` `:1644`, `_cts_from_breakout_event` (span `[start, confirmation]` `:1563-1591`, `confirmation <= apply <= D`) | backward / `<= D` | BOUNDED | |
| `_replay_step_no_patterns(i)` → `_update_active_range(i)`, `_maybe_update_cts_pre_confirm(i)`, `_maybe_confirm_cts_via_proximity(i)` | candle `i` + the resolver values | BOUNDED given L5 | |
| `_update_cycle0_data` → `has_unfilled_imbalance(df, lo, hi, check_to_idx=cts_idx)` | as-of `cts_idx` | EQUIVALENT for fills — **OUT** for instance existence / merged bounds at the bound (§2 residual) | |
| `_schedule_reversal_from_anchor` → `detect_best_for_anchor` + "apply within the watch window" `:829-835` | to `idx+5` | **FIX** L3 (detector) + BOUNDED (window `<= effective_end` after L4) | |
| per-candle step order `:578-581`: expiry before pending apply | not a read | — | the "apply exactly at `B` → false break" edge in §2 |
| `:1025-1036` debug print `self.df.iloc[i+1]` for `i in (102, 103)` | `i+1` | **FIX** (delete, §3.1.8) | print-only |
| `run()` terminal stamping `self.df.loc[first_rev_idx:, "market_state"] = "reversal"` `:462` | rows `> effective_end` | WRITE | forward stamp for chart/invariant consistency; every live caller's frame ends at the bound or discards the df. Unchanged. |
| `_rewind_to` replays from `i = 0` `:492`, ignores earlier jump requests `:497-500`, rebuilds `MarketStructureState` without `structure_id` `:485` | | OUT | LANDMINES "MarketStructure Deep-Couples…" point 1; identical in bounded and truncated runs. §8 |
| `unified_probe._run_phase1`: `find_true_first_breakout(bp, cur, upper=end_idx)`, `_select_extreme_retrace_candidate(df, est+1, end_idx)` `:195-216`, `_evaluate_reset_conditions(df, candidate)` `:219-258` | `[lo, hi]` explicit / single candle | BOUNDED (EQUIVALENT for the detector reads inside `_anchor_candidates` — all four detectors, `do_confirm=True`, `est > hi` dropped; both confirmation helpers return the FIRST qualifying candle, so bounding can only turn "confirmed at `k > hi`" into "unconfirmed", and both are dropped) | after §3.3 bounded by `bp` too |
| `unified_probe._bos0_inner_at_start(df, candidate)` `:342-353` (every reset, Phase 1 `:421` and Phase 2 `:634`) → `build_ad_hoc_bos0_reference_zone` → `identify_base_pattern` | to `candidate+5`, clamped to the frame — past `probe_end_idx` when the candidate is within 5 of it | **OUT** (L5b, §1/§8) | my original table said "backward" — wrong |
| `unified_probe._run_phase2`: `_make_market_structure(df_probe, end_idx=end_idx)` then reads `probe_events`, `df_probe["market_state"]` rows | | **FIX** via MS | the observed leaks |
| `structure_engine` main reversal probe: `unified_probe(df2.copy(), end_idx=reversal_apply_idx, enable_phase2=False)` `:288-296` | Phase 1 only | EQUIVALENT (above) — and L5b untouched, so its output cannot move | **H1 must stay byte-identical — the tripwire for this claim** |

---

## 5. Tests

### 5.1 New — the property test (`tests/test_ms_bounded_equals_truncated.py`)
The §2 definition, checked for **every** bound, so it cannot pass by fixture luck:
```python
raw = <ohlc rows>                                  # _make_reversing_data(), _make_uptrend_data(), + §5.2 fixtures
full = _prepare_df(raw)                            # tests/test_bounded_structure._prepare_df (:45): classification
                                                   #   + detect_patterns (pat*, is_range*) + compute_imbalance;
                                                   #   raw is a list of dict rows, so raw[:B+1] is a list slice and
                                                   #   _make_raw_df builds a fresh RangeIndex (MS requires it)
OUT_COLS = _OUT_INT_NEG1 + _OUT_INT_ZERO + _OUT_FLOAT_NEG1 + _OUT_FLOAT_NAN + _OUT_OBJ_EMPTY   # market_structure.py:36-51 (27 cols)
for B in range(start + 3, len(raw) - 1):
    bounded   = compute_bounded_structure(full, start, sd, end_idx=B, **mode)
    truncated = compute_bounded_structure(_prepare_df(raw[:B + 1]), start, sd, end_idx=None, **mode)
    assert ev_sig(bounded.events) == ev_sig(truncated.events)
    pd.testing.assert_frame_equal(bounded.df.iloc[0:B + 1][OUT_COLS].reset_index(drop=True),
                                  truncated.df.iloc[0:B + 1][OUT_COLS].reset_index(drop=True),
                                  check_dtype=False)                     # NaN-aware; rows < start too (rewinds write them)
    assert bounded.reversal_idx == truncated.reversal_idx
```
`ev_sig` = sorted `(type, idx, round(price, 6), json.dumps(meta, sort_keys=True, default=str))` — the
**whole** meta, so `expires_idx` on `REVERSAL_WATCH_START` / `REVERSAL_CANDIDATE` (`:723`, `:858`) is
compared (it is the field L4(a) makes equal; leaving it out would let L4(b) pass too). `mode` is
parametrised over `{}` and **scan mode** `{enforce_cts0_new_extreme: True, bos0_inner: float(full["l"].iloc[start])}`
(for `sd=+1`; the production Phase-2 caller is scan-mode and it diverges on its own axis — 22/57 bounds
at HEAD on the reversing fixture), and over `sd = ±1` and the fixtures. Runtime measured at HEAD:
`_prepare_df` ≈ 0.05 s, one bounded run ≈ 0.014 s, one fixture × one sd ≈ 3 s → **20–30 s** for the
full matrix; acceptable, mark it `@pytest.mark.slow` if the suite grows. `ms.debug=True` prints per step
(captured) and `:2529` emits a `FutureWarning` per run — noise only. **Run this test at the base first —
it must FAIL** (measured: reversing fixture 22 of 57 bounds, uptrend 38, first failure `B=3`: bounded
emits `CTS_UPDATED@4` — L1); cite that in the test's docstring. A difference caused by the
`attrs["imbalances"]` residual would surface here too — if it does, record it (§8) and decide with the
user whether it joins Plan A; do not silently exclude it.

### 5.2 New — boundary fixtures (each also fed to §5.1)
`_seg` (`test_bounded_structure.py:71-84`) cannot build these — it emits monotone segments with random
noise and no control of `candle_type` / big-candle flags. Hand-write OHLC rows against `CandleParams`
(`features/candle_params.py`: maru ≥ 0.65 body_pct, pinbar < 0.4, big flags = candle_len / max of the
prior 5 marus ≥ 0.5 / 0.65 — `candles_v2.py:169-250`) and the pattern rules (`structure_patterns.py:174-600`).
Budget real time for this; §5.1 is what proves the fix, these pin the mechanism so the failure message
names it:
- **L2 range at the bound:** a range candle `i` whose first confirming close is at `i+3`. `B = i+2` → no
  `RANGE_STARTED`; `B = i+3` → `RANGE_STARTED.idx == i+3`.
- **L1 pattern past the bound:** a breakout pattern anchored at `i` confirming at `i+2`. `B = i+1` → no
  `CTS_ESTABLISHED` for it; `B = i+2` → established.
- **L3 priority pre-emption:** an anchor `i` where a 2-candle pattern SUCCEEDs at `i+1` and a
  `continuous` (`:218-262`) SUCCEEDs at `i+2`. `B = i+1` → the 2-candle pattern establishes CTS at `i+1`
  (today: the `continuous` establishes it at `i+2`, past the bound). `B >= i+2` → `continuous` wins.
- **L4 watch at the bound:** a close-break of `bos_threshold` at `i` with a reversal pattern applying at
  `i+k`, `k <= 5`. `B = i+k` → `BOS_THRESHOLD_UPDATED(reason="probe_no_break")` at `B` and **no**
  reversal (expiry precedes the pending apply); `B > i+k` → reversal at `i+k` (unchanged); `B < i+k` →
  the anchor fails at `i` (`rv_anchor_failed`; no watch survives) — no events at `B`.
- **L5 BOS inner at the bound:** a `BOS_CONFIRMED` whose extreme `bos_idx` has an inside-bar pair at
  `bos_idx+3..+5` and a cycle-1 proximity confirmation that depends on that inner. `B = bos_idx+2` → the
  inner is derived without them (or is `None` → proximity off); `B >= bos_idx+5` → as today.
- **Negative test for the assert:** monkeypatch `MarketStructure._is_range_candle_given_confirm` to return
  the raw label (re-creating the L2 leak) on the L2 fixture and `pytest.raises(AssertionError)`. (Pushing
  `_effective_end` past the frame does NOT work — the loop dies with `IndexError` at `self._c[n]` before
  any event past the bound exists.)

### 5.3 Existing tests
| test | action |
|---|---|
| `test_pooled_structure_build::test_capped_run_emits_nothing_past_cap` (`:96`) | also assert `ev.idx <= cap` (already implied for every type but `BOS_CONFIRMED`; makes the guarantee explicit) |
| `test_pooled_structure_build::test_events_knowable_at_are_causal_in_end_idx` (`:82`), `::test_structure_columns_causal_over_window` (`:114`), `::test_project_to_window_matches_bounded_build` (`:149`) | the prefix-equivalence family (natural-end run vs `end_idx=cap`). Simulated post-fix on their fixture (cap = R−6): all stay green. Keep unchanged. **If any fails after Plan A, stop and report** — that is the documented §2 tail divergence on that fixture, and narrowing or deleting a test's claim is the user's call (Plan C repoints this module to the slicing builder anyway). |
| `test_bounded_structure::test_end_idx_caps_before_reversal` (`:173`), `::test_first_segment_events_match` (`:190`) | keep; expected green |
| `test_scenario3::test_short_data_with_end_idx_returns_finalized` (`:213-220`, `end_idx=10` on a 15-row frame through the legacy `compute_structure_scenario_3`) | runs a bounded MS on a longer frame → may change. Expected: still finalized; if the value moves, it is L1–L5 on a legacy path — record, don't chase |
| `test_unified_probe` | **no Phase-2 fixture exists anywhere in the suite** (`enable_phase2=True` appears only as a mocked kwarg in `test_first_trigger_migration.py:269`). Build the **multi-cycle synthetic fixture** (impulse → pullback that CONFIRMS → impulse, ≥ 3 CTS cycles; the existing fixtures establish exactly ONE) here — Plan B's §4 needs the same fixture and lands next — and assert on it that Phase 2's bounded MS emits no event past `end_idx`. |
| `test_sub_chain.py`, `test_first_trigger_migration.py`, `test_sub_structure_pool.py` | no change expected (sub builds slice to the bound). Any change there = the slice claim in §1 is wrong — investigate before proceeding |

---

## 6. Acceptance (`/compare` against the P1 revert save + chart review)

1. **H1: all CSVs byte-identical.** Main runs are `end_idx=None` (every edit is `n-1 → effective_end` or
   a fast-path `self.df`, equal there); the main reversal probe is Phase-1 only, equivalent, and L5b is
   untouched (§4). Any H1 delta = the equivalence claim is wrong → **stop and investigate** before
   looking at M15.
2. **M15: every delta must trace to a first_confluence probe Phase-2 outcome.** Run
   `debug/probe_fc_finalize.py` at the base and after the fix; diff the five rows. Predicted:

   | cycle | today | after Plan A | why |
   |---|---|---|---|
   | (0,0) | start 454, finalize 1020, `second_cts_reached` | unchanged | leaked candles 1722–1725 lie outside the retrace window `[459..783]`; the 2nd CTS_EST at 1020 is far from the bound 1721 |
   | (0,1) | 2365 / 2608 / `second_cts_reached` | unchanged | no leak observed; bound 2609 |
   | (1,0) | 2803 / **2844** / `no_retrace` | finalize **2843** (`no_retrace`, else-branch: cycle 0 no longer confirms inside the window) — or `second_cts_reached` if a 2nd CTS_EST exists ≤ 2843 (data-dependent); `starting_idx` **2803 may move** — the retrace window widens to `[2761..2843]` and a new most-extreme candidate at 2842/2843 could pass the reset | L1 |
   | (1,1) | 2915 / 3047 / `no_retrace` (else) | unchanged finalize (already the else-branch `end_idx`); events past 3047 gone | L2 |
   | (1,2) | 3304 / 3621 / `no_retrace` (else) | unchanged | no leak observed |

   A change in any row other than (1,0)'s finalize is possible only through L3 / L4 / L5 at that cycle's
   bound, or through a watch-expiry rewind in the old leaked tail that rebuilt earlier events. **Not
   pre-authorised: if it happens, stop, name the mechanism from the Phase-2 debug output (`ms.debug=True`
   prints `[POST_STEP]` / `[RV_EXPIRE]` / `[REWIND]`), and ask before accepting.** To make that possible
   the diagnostic must change first (§7): wrap `unified_probe._make_market_structure` to keep every
   Phase-2 `ms`, report `max(ev.idx)` over **all** iterations of a cycle (the last one alone hides earlier
   leaks), and stop filtering stdout to lines containing "unified_probe"/"WARNING" (`:102-108`) so the
   `[RV_EXPIRE]` / `[REWIND]` lines survive.
3. **Sub builds are byte-identical** except the sids whose probe row changed: under 3.2a the FC(1,0)
   sid (`2803/−1`, `inactive` at 3611 in the baseline) carries the finalize as `start_trigger_idx`
   (2844→2843 in `_sids.csv` / sid meta) and, if `starting_idx` moved, a different structure. Its KL /
   POI / fib / WVMI rows follow. Everything else in every M15 CSV: identical.
4. Log grep (`warning|skipping|unavailable|degenerate|pending|no sid`): no new lines.
5. Display the `=== Replay Timing ===` block. Expected: unchanged (the fix removes work; the L5 view
   slices only on probe frames).
6. **Pause for chart review** (the visible change is at most one `inactive` sub's outline), then
   `/commit-save`.
7. Anything else = regression.

---

## 7. Docs in the same commit
- **`LANDMINES.md`** "MarketStructure Range Look-Ahead Leaks Past `end_idx`": retitle "Bounded MS Runs
  Must Not Read Past `end_idx` (FIXED — Plan A, commit …)" **and update this plan's header citation**;
  rewrite the body to the §2 definition, the five sites L1–L5, the "clamp `D` alone is not enough —
  `RANGE_STARTED` is stamped at the label's `confirm_idx`" correction, the corrected watch mechanics
  (pending-apply-only; expiry before apply), the post-run assert as the guard, the §2 residual
  (`attrs["imbalances"]`) and L5b, and the "not prefix-equivalence" paragraph. Cross-link "Probe `end_idx`
  Is the Supreme Bound" and add L5b there as a known exception.
- **`structure/MARKET_STRUCTURE_SPEC.md`**: add a short "Bounded runs (`end_idx`)" subsection stating
  the definition, the shared 5-candle horizon (`range_max_k` = detector confirmation offset =
  `RangeLabelConfig.max_lookahead` = inside-bar scan half-width), the resolver view, and the assert.
- **`PART4_REFACTOR_SPEC.md`** §5 (`:~823` "all 3 once the MS bounds leak is fixed") → state the measured
  post-fix table; §17.11's Plan A line → "landed, commit …".
- **`GOTCHAS.md`** `:~1601-1606` (finalize native-vs-mapped entry) — still correct; add "fixed by Plan A"
  to its bounds-leak sentence.
- **Memory:** `project_sub_structure_pool_architecture.md` "PRE-EXISTING BUGS" → Plan A FIXED (commit);
  `reference_pool_redesign_groundtruth.md` FC table row (1,0) → measured values, drop the `*` footnote;
  `MEMORY.md` constraint line → fixed; Plan C §0 P1 wording if `starting_idx` 2803 moved.
- `debug/probe_fc_finalize.py`: the §6.2 changes (wrap `_make_market_structure`, per-cycle
  `max_event_idx` over all iterations, keep `[RV_EXPIRE]`/`[REWIND]` lines).

---

## 8. Findings outside scope (record, do not fix here)
- **L5b — the probe's ad-hoc BOS_0 derivation at a reset candidate** reads up to `candidate+5` past
  `probe_end_idx` (§1, §4). Not a live-causality issue (the probe's bound is a search bound; those candles
  exist at trigger time) but a reproducibility gap versus the stated bound, and closing it can move the
  H1 sid-1 start through the main reversal probe. Needs its own decision + `/compare`; recorded in
  LANDMINES "Probe `end_idx` Is the Supreme Bound" (§7).
- **`attrs["imbalances"]` at the bound** (§2 residual): instance existence needs `c3 = B+1`; merged-run
  bounds read `B+1`; consumers `_update_cycle0_data` and `_refresh_poi_inners_for_cycle`. Same family as
  `feedback_in_flight_vs_downstream_resolver`. If §5.1 trips on it, decide with the user.
- **`_rewind_to`** replays from candle 0 (`:492`), ignores earlier jump requests during the rebuild
  (`:497-500`), and rebuilds `MarketStructureState` **without `structure_id`** (`:485` — a rewind in an
  H1 sid ≥ 1 would stamp later events as sid 0; unobserved: 0 expiries on H1 in the reference window,
  15 / 3 on the M15 streams, all on slices). All in LANDMINES "MarketStructure Deep-Couples to Its
  Working DataFrame", point 1. Identical in bounded and truncated runs, so not a Plan A concern.
- `run()` `:411` returns `self.levels`, which is never defined → `AttributeError` when `start_idx >= n`
  (pre-existing; no live caller hits it — `_build_or_get_sub_geometry` guards `start_abs >= n`).
- **Prefix (clip) equivalence does not hold in the last 5 candles** (§2) — inherent; Plan C's model
  (geometry to the data edge, windows clipped from one run) is the answer, and `knowable_at_idx`'s
  half-clip of `CTS_ESTABLISHED` / `REVERSAL_CANDIDATE` remains a separate, known gap. The
  expiry-before-apply ordering (`:578-581`) is the sharpest instance (a reversal exactly at the edge is a
  false break).
- Terminal reversal stamping writes rows past the bound (WRITE, harmless for every live caller).

## 9. Definition of done
§5 tests green (the property test fails at the base — `B=3`, L1 — and passes after, in both modes);
§6.1–6.4 satisfied and every M15 delta named by mechanism, none accepted without asking; chart review;
§7 docs in the same commit; `/commit-save`; memory status updated; then Plan B.
