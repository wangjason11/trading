# Market Structure Spec — CTS/BOS/Range/Reversal (through Week 6)

Implementation: `MarketStructure` in `market_structure.py`.【fileciteturn1file0】

This document is meant to be the canonical behavior reference.

---

## Core principles

- **Sequential / event-driven**: process anchor `i`, decide what becomes known at or after `i`, then advance.
- **No skipping**: the engine always moves forward in time; it may “jump” the anchor index to the post-confirm candle for live-like timing; look-ahead windows (pattern confirmation, range labels) are evaluated offline by back-filling the candles
  before the decision candle — it never rewinds (the reversal-watch expiry's rewind was removed 2026-09-29, F3b).
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
    with `reconfirmed_idx` (the CTS_RECONFIRMED moment; `pb_reconfirm_idx` until
    Post-E·5, 2026-09-30) recorded — its `confirmed_idx` keeps the proximity moment
    (clamped up to the structure's lifecycle start, like every KL `confirmed_idx`).

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
  `expires_idx` — the frozen barrier holds only until then (the later-anchor cap, "Expiry inside a step" below).
- A breakout that establishes a NEW cycle ends the watch ("A new cycle ends an open watch" below).
- A reversal whose pattern confirms ON the expiry candle applies ("A reversal confirming on E applies" below).
If no reversal pattern from the close-break candle confirms within the window (`rv_anchor_failed`, decided at that
candle): BOS threshold updates to its wick extreme and the watch clears at once. (Originally the watch ran to its
expiry, BOS := the anchor wick, and execution rewound to anchor_idx+1【fileciteturn1file3】; since F3b, 2026-09-29, no
watch reaches its expiry, and the expiry + rewind were removed.)

### A new cycle ends an open watch (2026-09-29)
A watch freezes the CURRENT cycle's BOS. When a breakout establishes a new cycle (cycle >= 1) while a watch is open,
the market made a new extreme instead of reversing, and the new cycle's `BOS_CONFIRMED` supersedes the barrier the
watch froze. The watch ends at the establishing candle (`_end_watch_superseded_by_new_cycle`, called for that
`BOS_CONFIRMED`): its pending reversal — confirming only after that candle, not knowable yet — is dropped (its
`REVERSAL_CANDIDATE` stays in the stream unrealised), no rewind, and `bos_threshold`
is the new BOS. A later close beyond the NEW BOS opens a new watch by the normal rule. Trace: that `BOS_CONFIRMED`
carries `meta["ended_watch_pattern_anchor_idx"]` = the ended watch's close-break candle (= its `REVERSAL_WATCH_START`
idx; pattern realm, GLOSSARY "Naming Standard"; the key is present only when a watch was ended). No exception: a
pending that would confirm ON the establishing candle is dropped too (user decision after the landing review of
`2285232`, which built the case): on that candle the new BOS — the pullback low over a window that includes it — is
at or beyond its close, so that reversal could never have broken the current BOS (with the code as first landed the
cycle was established and then reversed on the superseded barrier on the same candle; at the watch's expiry candle
the expiry won instead and rewound the cycle away). It needs a `one_maru_opposite` breakout whose small opposite
candle also confirms the old pattern — a price gap, or a breakout candle ~20x the range-high-to-threshold distance;
0 in 20k random tails. The cycle-0 `BOS_CONFIRMED` cannot meet a watch (a watch needs a BOS; there is none before
cycle 0). A `CTS_UPDATED` inside a watch (same cycle, same BOS)
leaves it open. The watch was otherwise inconsistent with the cycle it guarded: the old pending applied on the
superseded barrier, sometimes on a close that never broke the new BOS (the old pattern's confirm threshold can lie
short of it), and invariant 4 raised (a live crash path); or the old watch's own expiry rewound the new cycle away
(`_make_double_rewind_data`: the .6066 high at 8 became neither a cycle nor a CTS update). Measured before (user
decision 2026-09-29): reference window 0 in-watch cycles (byte-identical); suite 38 (2 crash); random tails 0 crashes
in 48k but ~38% of trials with an in-watch cycle (all rewound away by the old watch's expiry); 32 crashes in 12k tails
started from an open watch. The options weighed (re-freeze the watch on the new BOS — identical to ending it on 12k
trials, since the new BOS is the pullback low at or beyond the old anchor's wick; block the breakout until the watch
resolves — the new extreme is lost as a CTS; keep and relax invariant 4 — a reversal without a break of the current
BOS): memory `project_zones_timing_audit_20260922.md`. Pins `tests/test_ms_new_cycle_ends_watch.py`.

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
A reversal — winner or pending — ends the watch AND clears the pending (`_apply_pattern_at_apply_idx`). Until
2026-09-29d a WINNER cleared only the watch, so its re-stepped apply row re-applied the pending when that confirmed on
the same candle (REVERSAL → REVERSAL, no event; the log's `[RV_APPLY]` line — every winner of the replay (5) and the
suite), and a pending confirming LATER than a later anchor's winner stayed set in the reversal row's
`pending_reversal_*` columns and kept `_should_stop_after_cts` from seeing a quiescent point — now, with a
`stop_after_cts_established` run past its N-th establishment, that reversal step can set `early_stop_idx` (read
only by `unified_probe`'s `[unified_probe phase2] early stop` print; no decision). Measured
(`reversal_shadow.py` winner-pending counters): 30k random tails 7,656 winners — 7,654 re-applied, 2 lingering (both in
the targeted stream; 0 with an early stop), 0 in the replay and the suite. Pins in `tests/test_ms_reversal_on_expiry.py`:
the `trace` taps record every reversal apply (P1 / P2: one), and the lingering case (seed 93 trial 3055).
Before the fix the step went on: the winner applied after the reversal (state left REVERSAL; e.g. a pullback's
`CTS_RECONFIRMED` + `STATE_CHANGED(reversal→pullback)`), a range finalized, and a later close-break could reverse
again — two `STATE_CHANGED(to=reversal)` for one sid, the H1 hand-off taking the first (df mask `.min()`) and every
reader the last (`compute_reversal_idx_by_sid`). 0 cases on the reference window (byte-identical); pins
`tests/test_ms_reversal_terminal.py`.

### Expiry inside a step — history; the later-anchor cap stays (2026-09-29)
**The later-anchor cap (live).** A reversal candidate against an open watch's frozen barrier at a later anchor `i`
(`A < i <= E`) must apply by the watch's E: `_best_bopb_pattern_at_anchor` caps it at `min(D, expires_idx)` — the
scheduler's rule (`_schedule_reversal_from_anchor`, where it never binds: the anchor-A pattern confirms by
`A + range_max_k`); apply `== E` is a winner (P2 below). It keeps a pattern completing after the window from
steering the step layout: uncapped, that candidate wins its step and its frozen back-fill runs straight to the watch's
pending apply p (<= E, F3b), skipping the anchors in between, so the reversal lands at p from the back-fill's frozen
state; capped, those anchors are stepped and the pending applies wherever p is stepped — in the P3 fixture, where the
capped anchor IS p, in that anchor's own non-frozen step (a `RANGE_UPDATED` + the range-side state change on the
reversal candle) — and an anchor in between could in principle carry an earlier reversal winner or a new-cycle breakout
that ends the watch. Measured 2026-09-29 (HEAD vs the cap removed): reference window 0 binds, byte-identical; suite
binds only in the P3 fixture (same output); 42k random tails: 376 signatures differ (208 of them in the 6k targeted
from-an-open-watch stream), the reversal candle always the same — no different reversal found, not proven impossible
(landing review of `221b420`). The cap stays. Pin
`tests/test_ms_reversal_on_expiry.py::test_the_reversal_candidate_is_capped_at_the_open_watchs_expiry`.

**History — the stop at the expiry (2026-09-29, removed the same day).** Until F3b the expiry ran before the pending
apply and fired exactly when the pending applied ON E, requesting a rewind to `A + 1` (false break: `bos_threshold` :=
the anchor's wick, `jump_to_idx` + a seed snapshot). The run loop honoured it only between steps, so a step that went
on after its own expiry had its work thrown away by the rewind + seed restore (a direct `state.state =`): a later
anchor's reversal winner applied after E and was discarded (F3), a second expiry inside the continuation overwrote the
jump target and the seed, and a `_rewind_to` rebuild — which ignored nested jumps (LANDMINES "MarketStructure
Deep-Couples…" point 1, history) — replayed a discarded continuation into a reversal and crashed on its assert. The
fix `52ac1e9` capped the later-anchor candidate at E (above) and ended every step at its expiry, returning the jump
target; `run()` asserted a rewind was never requested in REVERSAL. F3b (`c36777c`, next section) made the expiry
unreachable; the expiry, `_rewind_to` + the seed restore, the four stop returns, the run-loop rewind branch + its
tripwire and the rebuild assert were then removed (byte-identical). `_replay_step_no_patterns` asserts instead that a
watch is never open at its expiry candle — a watch left open would skip the BOS barrier step for good.

### A reversal confirming on E applies (F3b; 2026-09-29)
A close-break at A opens a watch expiring at `E = min(A + range_max_k, effective_end)`. `A + range_max_k` (5) is also
the LAST candle a reversal pattern anchored at A may confirm on (2-candle patterns: up to 4 candles after their end;
`continuous`: up to 3 — `detect_best_for_anchor`), and E is inclusive for the pattern rules, the scheduler
(`_schedule_reversal_from_anchor` drops only `apply > E`) and the later-anchor cap. **A reversal whose pattern
confirms ON E is a reversal, on every path:** in `_replay_step_no_patterns` the pending reversal applies BEFORE the
watch expiry check (user decision 2026-09-29, option I "E inclusive"). Before, the expiry ran first, and the same kind
of pattern confirming on E was
- (P1) applied when the close-break candle was its own step's anchor and its pattern won that step — the H1 main
  reversal of the reference window (sid 0 @902, 2026-01-09 12:00: `continuous` 897-899, 900/901 pinbars cannot
  confirm, 902 = E confirms) is one; 3 of the window's 5 reversals are P1 at E;
- (P2) applied when a later anchor's reversal winner applied at E (the winner path clears the watch first;
  `test_ms_reversal_on_expiry::test_p2_a_later_anchors_reversal_winner_on_the_expiry_applies`);
- (P3) DISCARDED as a false break when the pattern never won a step — the close-break candle ran inside another
  step, or its own step chose an earlier-applying winner — so it lived only as the pending: the expiry rewound to
  `A + 1` and re-ran those candles with a barrier decided at E (events stamped before the moment they became
  knowable), often reversing a few candles later, or never.
The outcome depended on the step layout, not the market. At E's close the watch's last candle closes and the pattern
confirms at the same moment — nothing past E is read either way. At the data edge (LANDMINES L4) the old rule was a
repaint: a false break at the bound B became a reversal AT B as soon as B + 1 arrived (the watch then expires later);
the new rule is prefix-stable. Every clear of the pending also ends the watch (the apply, a new cycle — before, also
the expiry and `_rewind_to`), so an open watch always holds a pending applying by E and **no expiry fires any more**:
the expiry, the rewind + seed restore and the "stop at the expiry" returns were removed in the next commit (a guard
asserts a watch is never open at its expiry candle; the later-anchor cap stays — "Expiry inside a step" above). Measured before (`reversal_shadow.py` /
`random_tail_search.py` F3b counters, scratch variants): reference window 0 expiries → byte-identical; suite 30
expiries (20 at the data edge); 42k random tails per tree: 813 P3 discards vs 1,179 P1 + 36 P2 applied, the rule
changes exactly the expiry trials (mostly none → a reversal at the edge; else a reversal 1–26 candles earlier, or 3
later where the old rewind had found a reversal retroactively), 0 errors, 0 expiries left. Rejected: "E exclusive"
(cap every reversal at E − 1) removed the H1 main reversal — sid 0 never reversing in the window — and 3 of the
window's 5 reversals; "a confirm on E counts as no pattern" lost the same reversals. Pins
`tests/test_ms_reversal_on_expiry.py` (P3 ×2, P1, P2, the edge's prefix stability, the cap).

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
`reconfirmed_idx` (the CTS_RECONFIRMED moment). Only main (H1) CTS zones are
exported: a sub's KL export is BOS-only, so a sub's upgrade stays internal
(the reference window's one CTS_RECONFIRMED, confluence sub 1 cycle 1 @1904).

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

## Look-ahead windows without rewinds

Range evaluation uses a look-ahead window (min/max K): a candle is marked a range-start candidate, later candles
decide range vs breakout, and the candles between the candidate and the decision are processed with the thresholds
known at the decision. MS does this in ONE forward pass: `_step_anchor` back-fills those candles offline
(`_replay_step_no_patterns(k, freeze_range=True)`) before acting at the decision candle, then continues sequentially
(pattern confirmation windows work the same way). It never rewinds: the original design's "seed snapshot → rewind to
the start → replay → resume at decision + 1" survived only as the reversal-watch expiry's rewind to `anchor + 1`,
unreachable since F3b and removed 2026-09-29 ("Expiry inside a step" above).

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
  (and, until its removal 2026-09-29, the expiry rewind target). A reversal pattern applying past the edge is
  never scheduled; one applying **exactly at** the edge reverses there on every
  path (a watch's pending applies by its expiry, and no watch is open past it; "A
  reversal confirming on E applies", F3b 2026-09-29 — before, the pending-apply path discarded it as a false
  break, which the next candle turned into a reversal at the old edge). Bounded ==
  truncated either way; one applying before the edge reverses as usual;
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
reversal — after the N-th `CTS_ESTABLISHED` in `events` (counted from the
event list, the same source the probe classifies from; no `structure_id`
filter — one instance runs one structure. Both date from when an expiry
rewind rebuilt the event list and reset `structure_id`; no rewind exists since
2026-09-29). `early_stop_idx` records
the first anchor NOT processed because of the stop (`None` = no early stop:
the count was never reached, or the N-th CTS landed on the last in-bound step
— then the run simply ends at its bound; "stopped early" is read from
`early_stop_idx`, never from the count). The check runs after `_step_anchor`
returns. Cost of "quiescent": a watch open at the N-th CTS resolves within its
window (its pending applies → reversal, the loop ends; or a new cycle ends it)
— a few extra candles, not asserted `<= 5`.

Semantics: the stopped run's events are an **ordered prefix** of the run without
the option (exact — until 2026-09-29 only when that run had no rewind after the stop point), and its
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
- reversal watch (invariant 4): an active row has a frozen barrier, and `bos_threshold` does not change between two
  consecutive rows of the SAME watch — same `reversal_bos_th_frozen` (2026-09-29). Back-to-back watches were two
  watches: an expiry rewound to anchor + 1 and that candle could open a new watch on the moved barrier, strictly
  beyond the old frozen one; comparing every consecutive active pair made that a false positive that ended the run
  (every measured false positive was of this kind; pins `tests/test_ms_invariant_watch_identity.py`). The shape is
  gone with the expiry (F3b, 2026-09-29). The
  guarded case — a BOS moving inside one watch — was reachable on real rows through a cycle established while a
  watch is open (GOTCHAS "A Cycle Cannot Be Established Inside an Open Reversal Watch") until that cycle started to
  END the watch ("A new cycle ends an open watch", 2026-09-29): the only `bos_threshold` write that can run during a
  watch is that `BOS_CONFIRMED`, so the check is now a pure tripwire (pinned on constructed rows).

Reversal is terminal (cannot leave reversal once entered) — asserted at the source (2026-09-28): `_set_state`
raises on any transition out of REVERSAL (a tripwire: "Reversal inside a back-fill" above stops every path that
reached one). (Until 2026-09-29 a `_rewind_to` rebuild asserted it never reached one and `run()` asserted a rewind
was never requested in REVERSAL — both removed with the rewind, F3b.) A reversal watch is never open at its expiry
candle — asserted in `_replay_step_no_patterns` (2026-09-29: its pending applies by then; a watch left open would
skip the BOS barrier step for good). (The former df-level check ran after `run()`'s forward stamp of `market_state` and could never fire
— deleted.)【fileciteturn1file11】

---

## Compute_structure variants (orchestration layer)

`MarketStructure` is the underlying state-machine engine. Two orchestration
functions wrap it (`structure/structure_engine.py`):

| Function | Start source | Multi-structure continuation | Use case |
|---|---|---|---|
| `compute_structure` | Scenario 1 (auto-identify via `identify_start_scenario_1`) | ✓ (each reversal's start via `unified_probe` + scan-from-start) | H1 main pipeline (orchestrator) |
| `compute_bounded_structure` | Caller-provided (a validated `unified_probe` start) | — (ONE structure; stops at its first reversal and reports `reversal_idx`) | Every sub build (the pool's geometry) + the per-sid scan-from-start runs |

**Per-reversal continuation (Step 4, 2026-06-20).** `compute_structure` (H1 main)
selects every post-reversal start via the **`unified_probe` + scan-from-start**
path — the SAME primitive the subordinate reversals use: reference = the prior
sid's most recent `{CONF/UPD/EST}` CTS; the probe runs in the flipped direction
over `[prior CTS anchor, reversal apply idx]` and hands back a DECISION (start +
BOS_0 inner), NOT events; the reversed structure's cycle-0 CTS_0 is then
established by a fresh **unbounded** scan-from-start MS run gated on that BOS_0
inner (`enforce_cts0_new_extreme` + `bos0_inner`).

**Deleted 2026-09-30 (user decision; no production caller, tests only):** the
legacy orchestrators `compute_structure_from_start` (caller-provided start +
Scenario 2 → Exception 1 → Exception 2 per reversal) and
`compute_structure_scenario_3` (the Scenario 3 BOS_0 probe, Phase 1, + an optional
multi-structure continuation, Phase 2), with `identify_start_scenario_2_after_reversal`
(Scenario 2 + its Exception 1) and their private helpers
(`_find_closest_candle_to_outer`, `_get_bos0_zone_bounds`, `_probe_reset_pips`,
`Scenario3Result`) and `tests/test_scenario3.py`. The Exception 2 probe and the
Scenario 3 BOS_0 probe went with them; `unified_probe` had replaced both (PART4
§4.4, memory `project_unified_identify_start_probe.md`). Git history keeps their
text (e.g. this section before the deletion commit).

---

## Probes

One probe primitive: `structure/unified_probe.py` (`unified_probe`) — for every
trigger type and the main reversals. Its contract is in the module docstring and
PART4 §4.4: a deterministic pass (the shared `find_true_first_breakout` routine +
the two-condition retrace reset, no MS) for every caller, plus an MS-based
iterative pass (Phase 2) for `first_confluence` only.

### Common probe patterns

- **Always on `df.copy()`** — no mutation of outer state until the result is accepted
- **Max iterations cap** (10) — prevents infinite loops; all iterations resetting → `max_iterations` (pending)
- **Retrace window opens at CTS_0's established MOMENT + 1** (Phase 1 `tfb.est_idx + 1`; Phase 2 `cts0_established_idx + 1` since Plan E E3c) — after the candle that set the level, so the CTS anchor candle (the breakout span's **pattern extreme**, at or before that moment) is never read: its far wick belongs to the breakout leg away from the zone, not a return to it (GOTCHAS "Exception Check Must Exclude CTS_ESTABLISHED Candle" — learned on the deleted Scenario 3 probe)
- **`probe_end_idx` is the supreme upper bound** for both the breakout search and the retrace window (LANDMINES "Probe `end_idx` Is the Supreme Bound")
- **Reset tolerance scales with timeframe** — `zones/zone_proximity.py::DEFAULT_PROBE_RESET_PIPS` (H1=3, M15=2.5, M5=2; `float` because M15 is fractional) + the wick cap `DEFAULT_PROBE_RESET_WICK`. Invariant: `DEFAULT_PROBE_RESET_PIPS[tf] < DEFAULT_PROXIMITY_PIPS[tf]` per TF (asserted at module load).

