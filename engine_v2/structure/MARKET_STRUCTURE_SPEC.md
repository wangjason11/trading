# Market Structure Spec — CTS/BOS/Range/Reversal (through Week 6)

Implementation: `MarketStructure` in `market_structure.py`.【fileciteturn1file0】

This document is meant to be the canonical behavior reference.

---

## Core principles

- **Sequential / event-driven**: process anchor `i`, decide what becomes known at or after `i`, then advance.
- **No skipping**: the engine always moves forward in time; it may “jump” the anchor index to the post-confirm candle for live-like timing, and it may **rewind + replay** only when needed to correct thresholds (primarily range evaluation).
- **Structure IDs**: structure_id partitions regimes; increments on terminal reversal. Zones and charts use structure_id to filter.

---

## State machine overview

MarketStructure maintains an internal state object (MarketStructureState) with:
- `struct_direction` (+1 / -1)
- `structure_id` (regime id)
- CTS lifecycle: `cts_cycle_id`, `cts`, `cts_threshold`, `cts_phase_debug`, `cycle_stage`
- BOS lifecycle: `bos` (the BOS anchor `Point(bos_anchor_idx, price)`, built from `_emit_bos_confirmed`'s `bos_anchor_idx` parameter, never from the emitted `ev.idx`; `bos_confirmed` before Plan E E2a), `bos_threshold`, `bos_event`
- Range: `range_active`, `range_hi`, `range_lo`, `range_start_idx`, `range_confirm_idx`
- Reversal watch: `reversal_watch_active`, `reversal_bos_th_frozen`, and pending reversal fields【fileciteturn1file3】

---

## Definitions

### CTS
A continuation level established within the current structure direction.
- It is confirmed by EITHER a pullback pattern OR sd zone proximity
  (whichever fires first — see "Dual CTS confirmation paths" below).
- CTS emits:
  - `CTS_ESTABLISHED` when a new CTS cycle begins (level anchored at an extreme, `meta["cts_anchor_idx"]`;
    `ev.idx` = the moment, the breakout's apply candle, since Plan E E4a).
  - `CTS_CONFIRMED` when CTS is confirmed by the first of pullback / proximity.
    `meta["confirmation_method"]` ∈ {`"pullback"`, `"sd_zone_proximity"`}.
  - `CTS_UPDATED` when the CTS anchor moves to a new extreme within the appropriate stage (rules depend on cycle stage).
    Pre-confirm, BOTH sources apply ONE rule — a STRICT new extreme beyond the current CTS in the structure
    direction (`MarketStructure._is_new_cts_extreme`; a tie keeps the first occurrence): the raw path (a candle's
    wick, `via="replay_raw"`) and the pattern path (a continuation breakout while the cycle is unconfirmed, its
    pattern extreme). A continuation breakout whose extreme does not clear the current CTS emits nothing, leaves
    `st.cts` and skips the POI-inner refresh — it still breaks the range, sets BREAKOUT and records
    `last_breakout_pat_apply_idx`. So the CTS never regresses (fixed 2026-09-27: the pattern path used to set
    `st.cts` unconditionally, and after a dip a lower breakout could pull the CTS back — then a candle between the
    two levels drew a spurious raw update and the pullback confirmed the wrong candle; pins
    `tests/test_ms_cts_update_no_regress.py`). Consequence: a pattern-path `CTS_UPDATED`'s anchor is always its
    apply candle — the breakout's span is `[anchor candle, apply]` and every span candle before the apply candle was raw-processed in the back-fill, so none lies beyond the current CTS (one that did became the CTS: a tie); only the apply candle, not yet raw-processed, can be a strict new extreme (asserted at the emit;
    ARCHITECTURE "`ev.idx` convention").
  - `CTS_RECONFIRMED` (new) when a valid pullback pattern fires AFTER CTS was
    already confirmed via proximity. The original CTS_CONFIRMED stays at the
    proximity idx; the CTS zone meta is upgraded to `confirmation_method = "pullback"`
    with `pb_reconfirm_idx` recorded.

### Cycle-0 pre-CTS_0 scan-from-start mode (`enforce_cts0_new_extreme`)

> **Repurposed 2026-06-07 (true-first-breakout cycle-0 redesign)** from the
> earlier partial anchor-extreme gate to the full pre-CTS_0 scan-from-start
> mode. Full design: `memory/project_true_first_breakout_cycle0.md`; probe-side
> detail: `PART4_REFACTOR_SPEC.md` §4.4.

`MarketStructure` accepts `enforce_cts0_new_extreme: bool = False` and
`bos0_inner: Optional[float] = None`. When the flag is True (the
**pre-CTS_0 scan mode**), `bos0_inner` is **REQUIRED** (raises otherwise):
while cycle 0 is unestablished MS delegates the entire breakout search to
the shared `find_true_first_breakout` routine (mechanism B against
`bos0_inner` + strict full-pattern new extreme + cycle-0 tie-break),
establishes CTS_0 at the located winner via its NORMAL cycle-0 path, then
resumes. Because the unified probe used the SAME routine with the SAME
`bos0_inner` to decide the start, MS re-finds the identical CTS_0 by
construction (no seed-and-resume). Cycles ≥ 1 are unaffected (subsequent
CTSes break the prior CTS by construction).

Used by `unified_probe` (deterministic method + phase 2) and by
`compute_bounded_structure` for every M15 sub (Commit 1). The same
mechanism is the planned wiring for the deferred main `sid=0 cycle=0` fix
(Commit 2). The old partial-gate helper `_cts0_new_extreme_passes` was
removed.

### BOS
A break level; confirmed by breakout logic.
- BOS emits `BOS_CONFIRMED` when confirmed.

### Range
A consolidation state bounded by (range_hi, range_lo).
- Can be started by explicit range candidate logic and/or by pullback rules (see below).
- Can expand (update hi/lo) based on later candles and based on pullback pattern window extremes.

### Reversal watch (frozen BOS barrier)
Starts **only** when a candle **close-breaks** the active BOS threshold.
During watch:
- BOS threshold must not update.
- Reversal candidates are detected relative to a frozen barrier and must apply by the watch's
  `expires_idx` — the frozen barrier holds only until then ("Expiry inside a step" below).
If no valid reversal pattern appears within watch window:
- BOS threshold updates to the close-break anchor candle wick extreme,
- watch clears,
- execution rewinds to anchor_idx+1 and proceeds normally.【fileciteturn1file3】

### Reversal inside a back-fill (terminal; 2026-09-28)
A scheduled (pending) reversal is applied by the per-candle step (`_replay_step_no_patterns`), which also runs
inside every **frozen back-fill** — candles a step processes offline around the candle it acts on: `_step_anchor`'s
winner back-fill `[anchor, apply)` and its no-winner range back-fill `[anchor, min(confirm, D))` (the step's
`D = min(anchor + range_max_k, effective_end)`), and a breakout's `_post_apply_range_check` back-fill
`[apply, min(confirm, apply + range_max_k))` — which starts AT the apply candle and runs past it. A reversal applied
there is the structure's end: **the step ends at that candle** — the rest of the back-fill, the winner's apply (a
later candle), the range finalize and the anchor's re-step are never reached, so the structure emits no event and
writes no row after its reversal candle. (It may have READ past it: the look-ahead to `D` chose the winner whose
back-fill reversed.)
A reversal WINNER (the anchor's own reversal pattern) applies at its apply candle and that candle's one per-candle
step still runs, but in REVERSAL the raw CTS update, the sd-zone proximity confirmation and the BOS barrier all skip
(`_maybe_update_cts_pre_confirm` / `_maybe_confirm_cts_via_proximity` / `_bos_barrier_step`); only the range update
still runs — it can emit `RANGE_UPDATED` (and a range-sync `CTS_THRESHOLD_UPDATED`) on the reversal candle (the
reference window's 5 reversals each have one `RANGE_UPDATED` there). Without the proximity skip an outside-bar anchor
(its new high moves the CTS to the anchor, so proximity first fires on the apply candle) confirmed the CTS and
created a range after the reversal (landing review F1).
Before the fix the step went on: the winner applied after the reversal (state left REVERSAL; e.g. a pullback's
`CTS_RECONFIRMED` + `STATE_CHANGED(reversal→pullback)`), a range finalized, and a later close-break could reverse
again — two `STATE_CHANGED(to=reversal)` for one sid, the H1 hand-off taking the first (df mask `.min()`) and every
reader the last (`compute_reversal_idx_by_sid`). 0 cases on the reference window (byte-identical); pins
`tests/test_ms_reversal_terminal.py`.

### Expiry inside a step (stop at the expiry; 2026-09-29)
A close-break at anchor A opens a watch that expires at `E = min(A + range_max_k, effective_end)` with a pending
reversal applying at `p <= E` (no pending → `rv_anchor_failed` clears the watch at once). The expiry runs before the
pending apply in the per-candle step, so it fires exactly when `p == E` (Open F3b below) and requests a rewind to
`A + 1` (false break: `bos_threshold` := the anchor's wick, `jump_to_idx` + a seed snapshot). Two rules keep a step
from acting past it:
- **The frozen barrier holds only until E.** A reversal candidate against it at a later anchor `i` (`A < i <= E`)
  must apply by E: `_best_bopb_pattern_at_anchor` caps it at `min(D, expires_idx)` — the scheduler's rule
  (`_schedule_reversal_from_anchor`, where it never binds: the anchor-A pattern confirms by `A + range_max_k`).
  Apply `== E` stays a winner: the winner path applies it before the apply candle's own step, whose expiry check
  then finds the watch cleared. Live-like: at E no reversal has completed, so the watch expires; a pattern that
  completes later re-qualifies against the post-expiry barrier after the rewind.
- **The step ends at the expiry.** An expiry requested inside a step — in the winner back-fill, in the breakout's
  `_post_apply_range_check` (which ends too; then `_step_anchor` skips the apply-row re-write), or in the no-winner
  range back-fill — ends the step at that candle, like a terminal reversal ("Reversal inside a back-fill"): no
  apply, no finalize, no re-step. The step returns the jump target: the run loop honours it; a `_rewind_to` rebuild,
  which ignores the request, resumes there without a reset as it always did (LANDMINES "MarketStructure
  Deep-Couples…" 1(a)).
Before the fix the step went on and the rewind + seed restore (a direct `state.state =`) threw the rest away: a later
anchor's reversal winner applied after E and was discarded (F3 — `tests/test_ms_expiry_stop.py`: the watch expires
at 9, the anchor-9 reversal applied at 10); a second expiry inside the continuation overwrote the jump target and the
seed (the df invariant "bos_threshold changed during reversal watch" fired); a rebuild replaying the continuation
re-applied the discarded reversal and crashed on its "never reaches a reversal" assert. `run()` now asserts a rewind
is never requested in REVERSAL. 0 expiries on the reference window (byte-identical).
**Open (F3b):** because the expiry precedes the pending apply, a reversal pattern that confirms ON the expiry candle
is discarded as a false break — and that is the only way the false-break rewind is reached (every expiry in the
suite, 42/42). LANDMINES L4 decided this for the data edge only; the general case is undecided.

---

## Range behavior (canonical rules)

### Range reset vs not reset
- **Breakout**: may reset/deactivate range (depending on whether range was active).
- **Pullback**: **never resets range**.

### Pullback creates range (Week 5 rule update)
If a valid pullback pattern occurs and no range is active, it *creates* a range:
- range starts from the confirmed CTS index (range_start_idx = CTS idx)
- bounds:
  - if struct_direction=+1:
    - range_hi = CTS price
    - range_lo = lowest low of the original pullback pattern window (excluding confirmation candle)
  - if struct_direction=-1:
    - range_lo = CTS price
    - range_hi = highest high of the original pullback pattern window (excluding confirmation candle)

If a range exists, pullback expands it to include the pullback pattern window extremes (does not reset).【fileciteturn2file5】

### Threshold syncing from range
When a range is active:
- `cts_threshold` mirrors the breakout bound:
  - sd=+1: cts_threshold = range_hi
  - sd=-1: cts_threshold = range_lo
- This sync is a first-class event source for zones (CTS threshold updates) and is emitted when cts_threshold changes due to range sync.【fileciteturn2file14】

---

## Dual CTS confirmation paths

After `CTS_ESTABLISHED`, the engine watches for both:
1. **A valid pullback pattern** (existing path, all cycles)
2. **First sd zone proximity hit** (uses BOS inner + active POI inners) — gated by **both**:
   - **Rule 1 (narrow-gap gate):** `|cts_price − bos_threshold| ≥ min_gap_threshold` (per-TF table)
   - **Cycle-0 carve-out:** `cts_cycle_id > 0` (BOS_0's "initial_prior_extreme" status makes it semantically suspect regardless of gap magnitude)

   Both gates must pass for sd-proximity to be eligible to confirm CTS. See "Rule 1: narrow-gap gate" and "Cycle-0 carve-out" sections below for the rationale of each.

### Rule 1: narrow-gap gate

The dual-CTS sd-proximity confirmation path requires `|cts_price −
bos_threshold| ≥ min_gap_threshold` at the candidate candle. The
per-TF threshold (`zones/zone_proximity.py::DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS`)
is:

| Timeframe | min_gap_pips |
|---|---|
| H1 | 50 |
| M15 | 30 |
| M5 | 15 |

**Why:** when the BOS-to-CTS gap is small (< 50 pips on H1), the sd
proximity buffer can overlap the CTS price level itself or fire on
trivially-small retracements — producing spurious confirmations and a
cascade of premature cycle transitions. The 50-pip floor ensures the
V/λ retracement pattern has room to develop before proximity can
substitute for pullback as a confirmation signal.

**Monotonicity:** the gap is monotonically non-decreasing within a
cycle (BOS extends in struct direction via probe; CTS extends via range
sync — both widen the gap). So a cycle can cross the threshold at most
once (narrow → wide), never the reverse. Once wide, the gate opens for
the rest of the cycle.

**Evaluation point:** uses `st.cts.price` (the current CTS anchor's price) and
`st.bos_threshold` (the running BOS level). No self-rescue concern in
the in-MarketStructure check: sd-direction wicks of the candidate
candle wick AWAY from CTS, so they can't extend `cts_threshold` this
candle (extensions happen on struct-dir wicks, the opposite
direction).

**Implication for CTS_RECONFIRMED:** since proximity can't confirm CTS
in narrow cycles, the only way a narrow cycle's CTS gets confirmed is
via pullback (or, in the mid-cycle-crossing case, via proximity AFTER
the gap widens). The `CTS_RECONFIRMED` event — which fires when
proximity confirmed first AND pullback later fires — therefore only
ever appears in wide cycles or after a mid-cycle crossing where
proximity confirmed post-crossing before pullback.

**Range-under-proximity-only path narrows scope similarly:** the
"Option B" range seeding (see below) only fires when proximity confirms
CTS, which after Rule 1 is restricted to wide cycles / post-crossing.

### Cycle-0 carve-out (proximity disabled)

The proximity confirmation path is **skipped on cycle 0** of every
structure_id. Cycle 0's `BOS_0` is the swing extreme that existed
*before* the structure began (`source: "initial_prior_extreme"`), so the
`BOS_0`–`CTS_0` gap is unconstrained — it can be arbitrarily small.
When that gap is smaller than ~2× `proximity_pips`, the sd-inner
proximity buffer overlaps the `CTS_0` price level and the very first
candle after `CTS_ESTABLISHED` trivially fires the trigger, producing a
premature `CTS_CONFIRMED` and a spurious cycle-1 `BOS_CONFIRMED` at the
narrow proximity-only retracement window. Cycles k>0 don't have this
pathology because `BOS_k` is the pullback extreme from cycle k-1, which
guarantees a structurally meaningful gap.

Cycle 0 therefore relies on **pullback confirmation only**. If no
pullback pattern fires before a same-direction breakout, the breakout
calls `_emit_cts_updated` (extending cycle 0's CTS to the new extreme —
only when its pattern extreme IS a strict new extreme, see "CTS" above),
not `_emit_cts_established` — keeping the same cycle alive rather than
spawning a phantom cycle 1.

The proximity check picks the closest-to-current-price sd inner across
BOS and POIs. POI inners are refreshed at `CTS_ESTABLISHED` (new cycle)
and at each `CTS_UPDATED` (CTS extended → Fib bounds expand → IC
candidates may shift). The snapshot is per-cycle in
`MarketStructureState.poi_inners_for_cycle`. Both refreshes still run
on cycle 0 (cheap; the cycle-0 POI refresh also keeps `cycle0_data` —
the Scenario-2 cond2 mirror — in sync via `_update_cycle0_data`), but
the per-candle check is gated off. The snapshot's ONLY reader is that
check (`_maybe_confirm_cts_via_proximity` → `_check_proximity_at_candle`).

**Snapshot vs per-candle — deliberate approximation:** The proximity
check uses a per-cycle POI snapshot, NOT a full per-candle activity check.
Refresh points are CTS_ESTABLISHED and CTS_UPDATED only — between those,
POI inners stay fixed. Trade-off: a POI whose IC qualification changes
between two CTS_UPDATED events (e.g., an imbalance fill mid-cycle that
disqualifies an IC candidate) is not reflected immediately by the
snapshot. In practice the deviation is small — POIs typically materialize
at CTS_ESTABLISHED time and don't shift much during the cycle. The
performance cost of full per-candle recomputation (Fib + IC scan + variant
selection) is substantial. The major refactor may revisit this.

**Snapshot vs per-candle — imbalance knowability is bounded by the consumer
gate, no refresh-time cut (Plan F, 2026-09-24):** MS's in-flight imbalance
reads — IC cond3 in `find_ic_candidates` over `(candidate, st.cts.idx]` and
Scenario-2 cond1 / cond3 in `select_fib_anchor_for_cycle` over `[BOS_1, CTS_1]`
/ `[BOS_0, CTS_0]` (both inside the resolver `compute_poi_inners_for_cycle`),
plus the cycle-0 mirror `_update_cycle0_data` over `[BOS_0, CTS_0]` — pass
`evaluated_at=None`: no c3 cut (IMBALANCE_FILL_SEMANTICS.md "Knowability — the
c3 rule"). Refreshed on `st.cts.idx` itself (a lag-0 `CTS_ESTABLISHED`, a raw
`CTS_UPDATED`), the snapshot can therefore count a gap whose c2 is the refresh
candle — one that forms only at the NEXT candle. The consumer gate makes that
safe: `_maybe_confirm_cts_via_proximity` is gated `i > st.cts.idx` and every
window ends at or before `st.cts.idx`, so every gap it counts (c2 ≤
`st.cts.idx`) has formed (`formed_at` ≤ `st.cts.idx + 1` ≤ `i`) by the candle
it is used on; `cycle0_data["has_unfilled"]` is read only at a later cycle-1
refresh (> CTS_0). **Do NOT add a refresh-time cut:** it would drop a gap that
is formed at every candle the snapshot is read on (and the snapshot stays fixed
until the next refresh), and it would break cond2 agreement with FibTracker's
equally uncut cycle-0 cache. FibTracker, which decides once per event, does cut
at the event's moment — the accepted MS/FibTracker divergence that follows (M1)
is documented in LANDMINES "Scenario 2 anchor agreement".

Whichever fires first confirms the CTS at that candle's idx:
- `CTS_CONFIRMED.meta["confirmation_method"] = "pullback"` if pullback won
  (or pullback fired at the same candle as the proximity wick)
- `CTS_CONFIRMED.meta["confirmation_method"] = "sd_zone_proximity"` if proximity won

If proximity confirmed first AND a valid pullback later fires, the engine
emits `CTS_RECONFIRMED` at the pullback idx. The original `CTS_CONFIRMED`
event is not modified (append-only contract). KL zone derivation post-pass
upgrades the CTS zone's `confirmation_method` to `"pullback"` and records
`pb_reconfirm_idx`.

### Range under proximity-only confirmation (Option B)

When CTS is confirmed via proximity (no pullback pattern fired yet), the
engine creates a range with bounds seeded by the proximity candle's wick:
- sd=+1: `range_hi = cts.price`, `range_lo = candle.low at proximity_idx`
- sd=-1: `range_lo = cts.price`, `range_hi = candle.high at proximity_idx`

This preserves all downstream behavior (CTS_THRESHOLD_UPDATED via range
sync, breakout detection via range thresholds) for proximity-confirmed
cycles. A subsequent pullback pattern (CTS_RECONFIRMED case) expands the
range as usual via `_ensure_range_on_pullback`.

### BOS_n+1 derivation rule

When transitioning to cycle n+1 via breakout pattern:
- **If cycle n had a pullback** (existing behavior):
  BOS_n+1 = pullback's deepest extreme in `[last_pullback_pat_apply_idx, breakout_apply_idx]`
- **If cycle n was proximity-only** (no pullback fired):
  BOS_n+1 = max retracement in `[cts_n_confirmed_idx, breakout_apply_idx]`

The implementation is in `_select_bos_on_breakout` and switches based on
`pullback_fired_for_cycle` state. Neither → it raises (unreachable: a cycle n+1
breakout needs CTS_n CONFIRMED, and both paths set `cts_confirmed_idx`; 2026-09-28 —
the silent fallback returned the structure's BOS_0). The BOS anchor is never after
its moment: `_emit_bos_confirmed` asserts `bos_anchor_idx <= idx` (the anchor-keyed
processing order needs the cycle's BOS before its CTS_ESTABLISHED; on the reference
window all 19 cycle >= 1 selections used the pullback window, none reversed). Both
guards — and the pattern-path `CTS_UPDATED` anchor == apply assert — fail the REPLAY,
on H1 and on subs alike: no MS caller catches `AssertionError` (a sub's geometry build
catches `ValueError` / `IndexError` only, LANDMINES "Lower-TF Pipeline Steps May Fail").

### Trigger threshold per timeframe

`MarketStructure.__init__` accepts `timeframe`, `proximity_pips`, and
`min_gap_pips` parameters. Both pip parameters fall back to per-TF
table lookups in `zones/zone_proximity.py` when None:

| Parameter | Source | H1 | M15 | M5 |
|---|---|---|---|---|
| `proximity_pips` | `DEFAULT_PROXIMITY_PIPS` | 9 | 6 | 3 |
| `min_gap_pips` (Rule 1) | `DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS` | 50 | 30 | 15 |

Pip size derived from `df.attrs["pair"]` (0.01 for JPY pairs, 0.0001
otherwise). The orchestrator wires both lookups via
`structure/structure_engine.py::_make_market_structure`.

The `min_gap_pips` value also feeds Rules 2 & 3 in the post-facto
`zones/zone_proximity.py::check_zone_proximity` scan — see that
module's docstring for narrow-cycle scan semantics.

---

## Pattern application timing

MarketStructure uses the notion of “apply candle”:
- If pattern is SUCCESS: apply_idx = end_idx
- If pattern is CONFIRMED: apply_idx = confirmation_idx【fileciteturn2file5】

The engine advances so that after a successful evaluation:
- it continues from **apply_idx + 1**, which corresponds to “the next candle that has opened but not closed yet” in live time.

---

## Rewind + replay

Range evaluation uses a lookahead window (min/max K) and may need to:
1) mark a candle as range start candidate
2) evaluate later candles to decide true range / breakout
3) correct break_thresholds for pattern evaluation between candidate start and decision

To do this deterministically:
- the engine stores a seed snapshot
- rewinds to range start
- replays forward with corrected thresholds
- returns to “decision point + 1”

This avoids “cheating” while still allowing batch computation.

---

## Bounded runs (`end_idx`)

A run with `end_idx=B` has **truncation semantics** (Plan A, 2026-09-19): it
produces exactly the events and output rows an unbounded run on the frame
truncated to `[0, B]` would (look-ahead labels recomputed on the truncated
frame). `effective_end = min(n-1, B)` is the run's **data edge** — nothing
past it is read, and the run does at the edge exactly what it does at the real
data edge. Unbounded runs (`end_idx=None`, the main H1 path) are the identity
case (`effective_end = n-1`).

What clamps at `effective_end` (never at `len(df) - 1`):
- the pattern-apply / range back-fill horizon `D = min(i + range_max_k, effective_end)`;
- the range label: a candidate whose `is_range_confirm_idx` (the FIRST confirming
  close in `[i+2, i+5]`, computed full-frame) is past the edge is *not* a range
  candle yet;
- the pattern detector: `BreakoutPatterns(df, end_idx=effective_end)` — candles
  past the edge do not exist for it (a SUCCESS / CONFIRMED that would need them is
  `None` / unconfirmed, never a candidate to be dropped later — which matters
  because `detect_best_for_anchor` returns ONE pattern per anchor by priority);
- the reversal watch: `expires_idx = min(anchor + range_max_k, effective_end)`
  and the expiry rewind target. A reversal pattern applying past the edge is
  never scheduled; one applying **exactly at** the edge is discarded as a false
  break (`probe_no_break`, rewind to `anchor + 1`) because expiry runs before the
  pending apply in the per-candle step — on the pending-apply path, i.e. when the
  close-break candle was processed inside `_replay_step_no_patterns`; a close-break
  candle that is itself an `_step_anchor` anchor has its reversal applied directly
  as the anchor's winner (apply ≤ `D`). Either way bounded == truncated. Note the
  expiry's own `BOS_THRESHOLD_UPDATED(probe_no_break)` does not survive the rewind
  (LANDMINES "MarketStructure Deep-Couples…" 1 (d)); one applying before the edge
  reverses as usual;
- the zone resolvers (BOS inner at `BOS_CONFIRMED`, POI inners at
  `CTS_ESTABLISHED` / `CTS_UPDATED`) read a view of the frame truncated at the
  edge (`_resolver_df()`, built once per run). `attrs["imbalances"]` stays
  full-frame — a documented residual (instance existence / merged bounds at the
  edge). Its **existence** half is unobservable (Plan F): a full-frame-only gap
  (c3 past the edge) is inside a resolver window only when its c2 == the edge ==
  `st.cts.idx`, and the snapshot is read only at `i > st.cts.idx`,
  `cycle0_data` only at a later cycle-1 refresh ("Snapshot vs per-candle"
  above) — candles the run never processes. The **merged-bounds**
  half matters only where a prefix and its merged run differ in degeneracy (the
  c3 rule's caveat) — LANDMINES "Bounded MS Runs Must Not Read Past `end_idx`",
  Known residual.

**Shared 5-candle horizon:** `range_max_k` = the detector's max confirmation
offset (`idx+5`) = `RangeLabelConfig.max_lookahead` = the inside-bar scan
half-width. One number; do not decouple them.

**Guard:** `run()` asserts post-loop that `max(ev.idx) <= effective_end`
(every emit site stamps `i`, an anchor, an extreme inside a pattern span ≤
apply ≤ `D`, or the range label's `confirm_idx`). If it fires, a forward read
was missed — fix the read, never the assert. Property test over every bound:
`tests/test_ms_bounded_equals_truncated.py`.

**Not prefix-equivalence.** A bounded run at `B` is NOT the natural-end run
clipped at `B`: the last `range_max_k` candles before `B` can legitimately
differ (a pattern the natural run is back-filling past `B` vs. those candles
processed as anchors; a reversal pending past `B`; a reversal applying exactly
at `B`). That is the 5-candle pending-confirmation nature of the machine.
Callers: `compute_bounded_structure` (subs — their frames are already sliced to
the bound), `unified_probe` Phase 2 (the first_confluence probe — the one
production path that runs MS on a frame longer than its bound).

## Early stop after N `CTS_ESTABLISHED` (`stop_after_cts_established`)

Opt-in (`MarketStructure(stop_after_cts_established=N)`, keyword-only, `N >= 1`,
default `None`; Plan B, 2026-09-20). The main loop also ends at the first
**quiescent** point — no reversal watch active, no pending (scheduled)
reversal, no pending rewind — after the N-th `CTS_ESTABLISHED` in `events`
(counted from the event list, the same source the probe classifies from;
`_rewind_to` rebuilds it, so no state counter is trusted and no `structure_id`
filter is applied — one instance runs one structure). `early_stop_idx` records
the first anchor NOT processed because of the stop (`None` = no early stop:
the count was never reached, or the N-th CTS landed on the last in-bound step
— then the run simply ends at its bound; "stopped early" is read from
`early_stop_idx`, never from the count). The check runs after `_step_anchor`
returns and after any pending rewind has been honoured (the rewind branch
`continue`s first), so the event list handed back is never one a rewind was
about to rewrite. Cost of "quiescent": a watch open at the N-th CTS resolves
within its window (apply → reversal, the loop ends anyway; expiry → rewind →
the stop lands after the next step) — a few extra candles, not asserted `<= 5`.

Semantics: the stopped run's events are an **ordered prefix** of the run without
the option (exact whenever that run has no rewind after the stop point), and its
output rows before `early_stop_idx` are that run's rows. The last step's range
back-fill may have stamped events/rows up to `range_max_k` past
`early_stop_idx` (a range candidate at the apply candle is stamped at its label
`confirm_idx`, exactly as in the unbounded run — FC(0,0) on the reference
window stops at 1021 with `RANGE_STARTED@1022`); rows from
`early_stop_idx + range_max_k` on are never written. The `end_idx` bound and
its post-run assert apply unchanged — the stop is in addition to the bound.
Only consumer: `unified_probe._run_phase2` (the first_confluence probe,
`N = 2`: the double-CTS rule is an early stop, finalize = the 2nd CTS's
`confirmed_at`). Tests: `tests/test_ms_stop_after_cts.py`.

---

## DF outputs (selected)

MarketStructure writes:
- `market_state` (labels)
- per-row event markers: `cts_event`, `bos_event`
- level info: `cts_idx`, `cts_price`, `bos_idx`, `bos_price`
- `range_active`, `range_hi`, `range_lo`, `range_start_idx`, `range_break_frac`
- `structure_id`, `struct_direction` per row【fileciteturn1file11】

---

## Invariants / guard checks

MarketStructure includes df-level invariant checks (low-noise):
- range_lo must not exceed range_hi while active
- CTS_CONFIRMED coherence with phase/stage
- BOS_CONFIRMED coherence

Reversal is terminal (cannot leave reversal once entered) — asserted at the source (2026-09-28): `_set_state`
raises on any transition out of REVERSAL (a tripwire: "Reversal inside a back-fill" above stops every path that
reached one), and a `_rewind_to` rebuild asserts it never reaches one — a KNOWN way it could: the rebuild ignores
nested expiry jumps (LANDMINES "MarketStructure Deep-Couples…" 1(a)), so its path can diverge from the first pass
(0 cases on the reference window and the suite; the one instance found since — F3 replayed by a rebuild — is fixed
by "Expiry inside a step"; the user chose a crash over the seed restore silently overwriting
the reversal). A rewind is never requested in REVERSAL — asserted in `run()` (2026-09-29; "Expiry inside a step":
a step ends at its own expiry, so the seed restore can never discard a reversal). (The former df-level check ran after `run()`'s forward stamp of `market_state` and could never fire
— deleted.)【fileciteturn1file11】

---

## Compute_structure variants (orchestration layer)

`MarketStructure` is the underlying state-machine engine. Three orchestration
functions wrap it for different start-identification strategies:

| Function | Initial start source | Phase 1 BOS_0 probe | Multi-structure continuation | Use case |
|---|---|---|---|---|
| `compute_structure` | Scenario 1 (auto-identify via `identify_start_scenario_1`) | — | ✓ (reversal start via `unified_probe` + scan-from-start) | H1 main pipeline (orchestrator) |
| `compute_structure_from_start` | Caller-provided | — | ✓ (legacy Scenario 2 + Exc1 + Exc2) | Legacy / tests — no production caller (subs use `compute_bounded_structure`) |
| `compute_structure_scenario_3` | Caller-provided + Phase 1 refinement | ✓ | ✓ if `run_continuation=True` (legacy Scenario 2 + Exc1 + Exc2; gated) | Subordinate probe across all multi-TF variants (counter and confluence; `run_continuation=False`); tests |

**Per-reversal continuation differs by function (Step 4, 2026-06-20).**
`compute_structure` (H1 main) now selects every post-reversal start via the
**`unified_probe` + scan-from-start** path — the SAME primitive the
subordinate reversals use: reference = the prior sid's most recent
`{CONF/UPD/EST}` CTS; the probe runs in the flipped direction over
`[prior CTS anchor, reversal apply idx]` and hands back a DECISION (start +
BOS_0 inner), NOT events; the reversed structure's cycle-0 CTS_0 is then
established by a fresh **unbounded** scan-from-start MS run gated on that BOS_0
inner (`enforce_cts0_new_extreme` + `bos0_inner`). This replaced the old
**Scenario 2 → Exception 1 → Exception 2** chain, which now survives only in
`compute_structure_from_start` (no production caller) and
`compute_structure_scenario_3` Phase 2 (test-only).
`identify_start_scenario_2_after_reversal` (incl. its Exception 1) is therefore
no longer reached from the main pipeline; deletion is deferred until those two
legacy callers are retired. The three functions otherwise still differ in
**how the very first start_idx is determined**.

### Starting-point rigor hierarchy

```
Lowest:  Caller picks start, no validation
         → compute_structure_from_start
Medium:  Auto-identify via Scenario 1 (lookback search)
         → compute_structure
Highest: Caller picks candidate + iterative probe validates/refines
         → compute_structure_scenario_3 (Phase 1)
```

---

## Probes

Two distinct probe mechanics exist in the structure layer. They share
common patterns (run on `df.copy()`, iterative with max cap, scan window
starts at `CTS_EST + 1`) but answer different questions.

### Probe type 1 — Exception 2 probe

**Question:** "Is the next-structure start (chosen by Exception 1) actually
a structural start, or just a pullback candle?"

**Where used:** Per reversal in the LEGACY continuation only —
`compute_structure_from_start` (no production caller) and
`compute_structure_scenario_3` Phase 2 (test-only). **NOT** in
`compute_structure` (H1 main), which migrated to the `unified_probe` +
scan-from-start reversal handoff in Step 4 (2026-06-20).

**Bounds:** `[exc2_candidate, reversal_confirmed_idx]` — bounded probe.

**Mechanics:** Run MarketStructure from candidate up to reversal. If a
CTS_EST fires in the probe AND any candle between `CTS_EST + 1` and
`reversal_confirmed_idx` reaches the prior CTS zone (within 10 pip
tolerance), exception triggers. New candidate = the reach-back candle
(strictly **later** than the previous candidate). Iterate up to 10 times.

**Outcomes:**
- **Exception triggered (any iter):** discard ALL probes, use settled
  candidate as start_idx, run **unbounded** MarketStructure from there in
  the outer loop.
- **No exception ever:** keep first probe's data (events, levels, df),
  continue outer loop from `reversal_confirmed_idx + 1`. Sid N+1 ends up
  split across two MarketStructure invocations — the bounded probe
  (pre-reversal portion) and the post-reversal continuation.

No status field — outcome encoded as boolean `exc2_triggered`.

### Probe type 2 — Scenario 3 BOS_0 probe (Phase 1 of `compute_structure_scenario_3`)

**Question:** "Is the arbitrary candidate start a true cycle-0 point, or
is it part of an older still-extending structure?"

**Where used:** `compute_structure_scenario_3` Phase 1. In production
invoked by `_run_subordinate_probe` (all multi-TF use cases — first/
subsequent counter and confluence) with `run_continuation=False`.

**Bounds:** `[start_idx, end_idx]` — `end_idx` optional.

**Mechanics:** Run MarketStructure from candidate. After ≥1 `CTS_EST`
fires AND the check window can be bounded (via `end_idx` or
`ef.cts_anchor_idx(cts_est[1])`), check if any candle in
`[ef.cts_anchor_idx(cts_est[0]) + 1, exc_upper]` reaches the BOS_0 zone inner bound
(within `pip_tolerance_pips`). If so, restart from that reach-back
candle (later than current). `original_bos0_bounds` captured at iteration
0 only and preserved across iterations.

**Exception check window:**
- Lower bound: `ef.cts_anchor_idx(cts_est[0]) + 1` — excludes the CTS anchor candle, i.e. the
  breakout span's **pattern extreme** (not a pullback candle; and the anchor, not the established
  moment `meta["confirmed_at"]` = `ev.idx` since Plan E E4a, though the two usually coincide —
  `ARCHITECTURE.md` "`ev.idx` convention"; a test-only path that keeps the anchor, PLAN_E Q10). That candle is still part of the breakout leg away from the zone, so its far wick
  is not a return to it (GOTCHAS "Exception Check Must Exclude CTS_ESTABLISHED Candle")
- Upper bound (`exc_upper`):
  - `end_idx` when defined (supersedes `cts_est[1]` per LANDMINES "Probe
    `end_idx` Is the Supreme Bound" — `end_idx` is a caller-defined hard
    bound; inner rules like "2 CTS_EST" don't narrow it)
  - else the second CTS_EST's anchor `ef.cts_anchor_idx(cts_est[1])` when ≥2 CTS_EST exist (live-mode fallback)
  - else no upper bound → pending

**Status field — conditions:**

| Condition | Trigger | Status |
|---|---|---|
| 1 | ≥1 CTS_EST + no exception found in `[cts_est[0]+1, exc_upper]` | finalized |
| 2 | Exception triggered (within bounded check window) | (loop continues — no status) |
| 3 | Reversal in probe before 2nd CTS_EST | finalized |
| 4a | No CTS_EST in probe (or no BOS_0 bounds) AND `end_idx is not None` | finalized |
| 4b | No CTS_EST in probe (or no BOS_0 bounds) AND `end_idx is None` | **pending** |
| 4c | 1 CTS_EST + `end_idx is None` (no way to bound check) | **pending** |
| (max iter) | 10 iterations all triggered exception | pending |

**Pending semantics:** Caller may re-invoke with same or advanced
`start_idx` once more data arrives. `_run_subordinate_probe` returns
`None` on pending so the sub isn't built for that trigger. Pending path
is dormant in current backtest (all triggers pass `end_idx` definitively
— CTS_CONFIRMED idx for first_confluence, first sd zone-proximity
trigger candle for first_counter, etc.).

### Phase 1 vs Phase 2 (Scenario 3 only)

- **Phase 1** = the BOS_0 probe loop above. Validates `start_idx`.
  Returns `Scenario3Result` with status, validated start, probe data.
- **Phase 2** = multi-structure continuation from the validated start —
  identical mechanics to `compute_structure`'s post-reversal handling
  (Exception 2 probe per reversal). Gated by `status == "finalized" AND
  run_continuation=True`.

**Currently unused in production:** Phase 2 is exercised only by
`tests/test_scenario3.py`. All production callers of
`compute_structure_scenario_3` pass `run_continuation=False` (only
`_run_subordinate_probe` calls it, probe-only). Worth knowing if a future
feature needs multi-structure continuation from an arbitrary validated
start — the path exists.

### Common probe patterns

- **Always on `df.copy()`** — no mutation of outer state until result accepted
- **Max iterations cap** (10) — prevents infinite loops
- **`CTS_EST + 1` scan window start** (`structure_engine.py` Scenario 3 Phase 1 probe, `compute_structure_from_start` and Scenario 3 Phase 2 Exception 2) — excludes the CTS anchor candle (`ef.cts_anchor_idx`; `CTS_ESTABLISHED.idx` until Plan E E4a) = the breakout span's **pattern extreme**, still part of the breakout leg away from the zone: its far wick is not a return to the zone, and including it caused false exceptions. It is not a pullback candle, and it is keyed on the extreme, not the established moment `meta["confirmed_at"]` (the two usually coincide; `ARCHITECTURE.md` "`ev.idx` convention"; GOTCHAS "Exception Check Must Exclude CTS_ESTABLISHED Candle")
- **Pip tolerance scales with timeframe** — values from `zones/zone_proximity.py::DEFAULT_PROBE_RESET_PIPS` (H1=3, M15=2.5, M5=2; type is `float` because M15 is fractional). Used by `compute_structure_scenario_3` Phase 1 probe AND by the legacy Exception 2 probes (`compute_structure_from_start` + `compute_structure_scenario_3` Phase 2; `compute_structure` no longer runs Exception 2 after Step 4 — its H1-main reversal handoff uses the `unified_probe` reset tolerances from the same table). Invariant: `DEFAULT_PROBE_RESET_PIPS[tf] < DEFAULT_PROXIMITY_PIPS[tf]` per TF (asserted at module load).【fileciteturn1file11】

