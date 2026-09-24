# Cross-Cycle Fib Unification — Design & Implementation Plan

> **Status:** DESIGN (started 2026-06-16) — **not yet implemented.** This is the
> first canonical spec for the cross-cycle Fib *decision logic* (anchor selection,
> dead-cycle walk, versioning). Until now the only description was the memory entry
> `project_cross_cycle_fib_mode.md` (Path B) + scattered docstrings. When this lands
> it **absorbs** that memory entry as ground truth.
>
> **Companion spec:** `FIB_LIFECYCLE_SPEC.md` covers the orthogonal *lifecycle* axes
> (active/ended/status, `start_idx`/`end_idx`, the §15 scalar model). This spec is
> about WHICH ANCHORS a cross fib uses and HOW FAR it spans — not its lifecycle
> projection. The two compose; neither replaces the other.
>
> **Line numbers** are 2026-06-16 hints; anchor on method/field names (stable).

---

## 0. One-sentence summary

Two implementations compute the same conceptual thing — "a fib whose anchor reaches
back across one or more prior cycles when those cycles still hold unfilled
sd-imbalances" — and have drifted apart: **Path A** (H1 main, post-reversal
Scenario 1/2/3) does a fixed **cycle-0→cycle-1** cross via three static imbalance
conditions; **Path B** (subordinate `cross_cycle` mode) does a **generalized
earliest_x→target** cross via a dead-cycle backward walk with integer versioning.
This spec **unifies the decision logic onto one shared pure routine** and, on that
unified base, **extends main's cross beyond cycle 1** up to a bounded ceiling.

---

## 1. Why this exists

Divergent dual implementations of one decision are a known latent-bug class here
(see `feedback_in_flight_vs_downstream_resolver`, `feedback_single_source_of_truth`).
Every fib-touching change deepens the A/B split. Two coupled goals:

1. **Unify** the cross-cycle *requirements* logic (unfilled-imbalance test,
   deepest-retrace / running-extreme anchor, the cross-cycle / dead-cycle walk, and
   versioning/logging) into ONE shared routine both modes call — extending the
   existing partial-share `select_fib_anchor_for_cycle`.
2. **Update** the post-reversal (main) cross to **extend beyond cycle 1** — a
   *bounded* multi-cycle cross — landed ON the unified base (the update folds into
   the unification, not a separate copy that gets re-touched).

**Prerequisite for true-first-breakout Step 4** (main reversals, H1 sid≥1), which is
post-reversal-fib-heavy — unify before Step 4 so Step 4 builds on settled logic. See
`project_true_first_breakout_cycle0.md`, `project_cross_cycle_fib_unification.md`.

---

## 2. Status quo (what exists today)

Both paths live in `FibTracker` (`zones/fib_tracker.py`), selected by
`fib_mode ∈ {"h1", "cross_cycle"}` at construction (orchestrator passes it). The
same orchestrator event loop drives both (`orchestrator.py:204-236`,
`on_cts_established / on_cts_updated / on_cts_confirmed / on_cts_threshold_updated`);
the branch happens *inside* each handler.

### 2.1 Path A — H1 main, post-reversal (Scenario 1/2/3)

- **Scope:** `sid ≥ 1` only, and **only the cycle-0 → cycle-1 cross**. `sid == 0`
  never crosses ("simple flow"). Cycles 2+ get plain single fibs.
- **State:** `self._scenario1[sid] ∈ {None, True, False}`. Scenario 1 decided at
  CTS_0 EST/UPD (`_handle_cycle0_scenario1:701`): `CTS_0_idx >= reversal_confirmed_idx`
  → TRUE (cycle-0 single fib unlocked); resolves FALSE at CTS_0 CONFIRMED if never
  reached. Cycle-0 data cached as a plain dict in `_cross_cycle_data[sid]["cycle0"]`.
  Its `has_unfilled` is stored **uncut** by the c3 knowability rule: it is cond2,
  judged at its use (CTS_1 EST, when every gap in `[BOS_0, CTS_0]` has formed), which
  keeps it equal to the MS in-flight mirror `cycle0_data`. The Scenario-1 activation
  decided at CTS_0 EST/UPD asks at that event's moment instead (the cut check at EST,
  `_c0_has_unfilled_now` on updates) — Plan F 2026-09-24; `IMBALANCE_FILL_SEMANTICS.md`
  "Knowability — the c3 rule".
- **Cycle-1 decision** (`_handle_cycle1_scenarios:761`): if S1 TRUE → **revert check**
  (`_should_revert_scenario1:1725` — BOS_1 reaches into prev structure's last BOS
  zone outer → flip S1 FALSE, kill cycle-0 fib via `_deactivate_cycle0_fib`, terminal
  `scenario1_revert`). If S1 (post-revert) TRUE → normal cycle-1 single fib. If S1
  FALSE → **Scenario 2 vs 3** via `select_fib_anchor_for_cycle` (3 static imbalance
  conds: cond1 cycle-1 unfilled, cond2 cycle-0 unfilled, cond3 BOS_1 hasn't filled
  cycle-0). All true → cross BOS_0→CTS_1 (`scenario_2_cross`); else single (Scenario 3).
- **Storage:** named slots in `_cross_cycle_data[sid]` (`cycle0` dict,
  `normal_cycle1` FibState, `cross_cycle` FibState); only the active winner mirrors
  to `_fibs[(sid,1)]`. **≤2 anchor candidates** (BOS_0 cross vs BOS_1 normal), no
  integer version. Update toggles cross↔normal (`_update_cycle1_fibs:1403`).
- **Shared util already used:** `select_fib_anchor_for_cycle` (`fib_tracker.py:99`),
  a pure function ALSO called by the MS in-flight POI resolver
  (`market_structure.py:1928`) so in-flight and downstream agree on Scenario 2. It
  covers **cycle 1 only** and only the S2-vs-S3 decision (not S1, not revert, not a
  walk). (Since Plan F, 2026-09-24, they agree on cond2 / cond3 but not always on
  cond1: FibTracker passes `evaluated_at` = the CTS_1 moment, the in-flight resolver
  `None` — the accepted M1 divergence, LANDMINES "Scenario 2 anchor agreement".)

### 2.2 Path B — subordinate `cross_cycle` mode

- **Scope:** all subordinate variants, every cycle, generalized cross.
- **Phase machine** `_m15_phase[(sid,cycle)] ∈ {pre_established, established, confirmed}`.
  `pre_established` (entered at CTS_{n-1} CONFIRMED) is driven by
  `CTS_THRESHOLD_UPDATED` (`on_cts_threshold_updated:1880`): anchor = running extreme
  past CTS_n (`_running_extreme_anchor:1803`), own-imbalance start = prospective
  BOS_{n+1} (`_find_prospective_bos:1773`, deepest pullback since CTS_n CONFIRMED).
- **Cross check** (`_m15_cross_check:1941`): (1) target cycle's own imbalance — since
  Plan F (2026-09-24) only a gap **formed** by the handled event's moment counts (§3
  item 1); if none, deactivate cross (no pre-established fallback). An own gap whose
  c2 is the `CTS_THRESHOLD_UPDATED` candle itself has not formed there, and with no
  later re-check the pre-established cross is never created (reference window: the
  cross fib counter sub 5 used to pre-create at 3806 for a cycle 1 that never
  establishes — sub 5 only ever establishes cycle 0 — and its IC 3654 twin POI no
  longer exist; `IMBALANCE_FILL_SEMANTICS.md` "Decided at the event"). (2) **dead-cycle
  backward walk** from target-1 to 0 using `_dead_cycles[sid]` cache (Interpretation
  B: cycle k dead when `[BOS_k,CTS_k]` has no unfilled sd-imbalance with fill checked
  **to the current candle**; each walked window ends at `CTS_k`, before the moment, so
  the knowability cut cannot change it); finds `earliest_x`. (3) version transition
  keyed by **start anchor**: new cross = v0; CTS extend = same version in place; start-anchor shrink (`earliest_x`
  moved up) = `cross_shortened` + v+1; same-anchor revival = in-place reactivate;
  `earliest_x == target` (no eligible prior) = `cross_failed`, established-phase falls
  back to single.
- **Storage:** ALL versions in `_fibs` as `(sid,cycle,"cross",v)` + a `(sid,cycle)`
  single fallback. Integer `_cross_version`. Monotonic supersede. Up to **N** versions.
- **Cycle 0:** single fib only, no cross, no pre-established. Late-activate on
  CTS_UPDATED (`18a6b32`, `_handle_cross_cycle_cts_updated:1098`) — since Plan F asked
  at the update's moment (a raw update's `idx`; a pattern-path update records none →
  no cut), so when the only gap has its c2 at the moment of an EST or raw update, the
  fib can first activate at the next raw update instead (reference window: conf sub 2
  and sub 3, cycle 0, one candle later).

### 2.3 Divergence map

| Dimension | A — h1 post-reversal | B — `cross_cycle` |
|---|---|---|
| Cross span | fixed **cycle 0 → cycle 1** | generalized **earliest_x → target** |
| Eligibility | 3 static imbalance conds @ CTS_1 EST | per-candle dead-cycle walk + cache |
| Anchor selection | `select_fib_anchor_for_cycle` (cycle-1 only) | `_m15_cross_check` internal |
| Phase/state | `_scenario1[sid]` + revert | `_m15_phase` 3-phase + `CTS_THRESHOLD_UPDATED` |
| Versioning | named slots, ≤2, toggle | integer `_cross_version`, ≤N, monotonic |
| Storage | `_cross_cycle_data[sid]` scratch; winner→`_fibs` | all versions in `_fibs` |
| "Clean reversal" notion | Scenario 1 + revert | none (subs have no S1) |
| Pre-established | none | yes (subordinate-only, §6 of lifecycle spec) |

**Key insight:** A's cond1/cond2/cond3 ARE a **single-step** (earliest_x∈{0,1},
target=1) specialization of B's dead-cycle walk. A's Scenario-1/revert are a
genuinely main-only OUTER GATE. B's pre-established has no analog in A and main does
NOT adopt it.

---

## 3. Unified model — one shared cross-requirements routine

Extract the cross-cycle *requirements computation* from Path B's `_m15_cross_check`
into ONE pure routine both modes call (extends the `select_fib_anchor_for_cycle`
pattern). The routine computes, for a given target cycle, the cross anchor decision
from the per-cycle BOS/CTS geometry + imbalance state. The four unified ingredients
(the user's stated scope):

1. **Unfilled-imbalance test** — sd-direction `has_unfilled_imbalance` over a cycle's
   `[BOS_k, CTS_k]` range with fill checked to the current candle (Interpretation B),
   counting only gaps **formed** by the moment of the decision. Two as-ofs (Plan F,
   2026-09-24; canonical: `IMBALANCE_FILL_SEMANTICS.md` "Knowability — the c3 rule"):
   the **fill horizon** (`fill_horizon_idx`; the snapshot walk's
   `snapshot_horizon_idx` — split from the window ends `own_window_end_idx` /
   `own_imb_start` in Plan E E2b) is a moment only on
   `CTS_THRESHOLD_UPDATED` and a raw `CTS_UPDATED` — on `CTS_ESTABLISHED` and a
   pattern-path `CTS_UPDATED` it is the CTS anchor until Plan E E3a; the **moment** is the routine's
   keyword-only, required `evaluated_at` (FibTracker: `event_moment` of the handled
   event; the MS in-flight resolver: `None` = no cut). Plan E E3a keeps ONE moment
   parameter (`evaluated_at`).
2. **Deepest-retrace / running-extreme anchor** — the CTS-side anchor (running
   extreme) and, for pre-established only (sub), the prospective-BOS start.
3. **Cross-cycle (dead-cycle) walk** — backward from `target-1` to 0, stopping at the
   first dead cycle, yielding `earliest_x` (the cross start cycle).
4. **Versioning / logging** — the start-anchor-keyed version identity (v0 / extend /
   shrink / revive) and its lifecycle logging.

**The shared routine is mode-agnostic for ingredients 1-3.** Mode differences become
*parameters*, not separate code paths:

| Parameter | h1 (main) | cross_cycle (sub) |
|---|---|---|
| `target_ceiling` (§4) | `M` (first cycle clearing prev-BOS-outer) | `None` (unbounded) |
| `fill_as_of` policy (§3.1) | **snapshot** (per-cycle frozen point) | **current** (running candle) |
| pre-established eval | OFF (no `CTS_THRESHOLD_UPDATED` cross) | ON |
| outer gate | Scenario-1 + revert (§5) | none |

### 3.1 The fill-as-of difference — A and B are NOT identical at `target=1` (decided 2026-06-16)

A's cond1/cond2/cond3 are **structurally** B's single-step walk (`earliest_x∈{0,1}`,
`target=1`): A's "Scenario-2 cross" == B's `earliest_x=0`; A's "Scenario-3 single" ==
B's `cross_failed→single`. **But they evaluate the cycle-0 liveness fill at DIFFERENT
candles:**

| Check | Path A (main, today) | Path B walk (sub) |
|---|---|---|
| cycle-1 own imbalance `[BOS_1,CTS_1]` | as-of **CTS_1** | as-of **current (CTS_1)** — same |
| cycle-0 liveness `[BOS_0,CTS_0]` | cond2 as-of **CTS_0** AND cond3 as-of **BOS_1** (frozen snapshots) | one check as-of **current candle** |

Fills are **monotonic** + CTS_0 < BOS_1 < CTS_1, so B's as-of-current is the strictest
point: **{B creates cross} ⟹ {A creates cross}**, not the reverse. A is a *superset* —
it creates/keeps a cross-cycle fib even when cycle-0's imbalance fills in the
BOS_1→CTS_1 window (the cross-cycle POI lingers); B drops it. This difference persists
through the fib's life (A re-uses the frozen snapshot on updates; B re-checks at
current).

**DECISION (user 2026-06-16): PARAMETERIZE the fill-as-of point; preserve both
behaviors.** The routine takes a `fill_as_of` policy: main passes its snapshot points
(cond2@CTS_0, cond3@BOS_1), subs pass "current." This keeps the historical A-vs-B
behavior intact so **11a is byte-identical for BOTH modes**. Whether main *should*
eventually converge on B's "current" semantics is a **separate, deferred, intentional-
diff** question (§13) — NOT folded into the refactor.

**Sanity floor (do NOT regress, §11a pass condition):** the unified routine with main's
parameters (`target=1`, `fill_as_of=snapshot`) MUST reproduce today's Scenario-2/3
decision **byte-identically** — including the frozen-snapshot superset behavior above.

---

## 4. Main-only — the target-cycle CEILING `M`

This is the **new capability** (the "update" task): main's cross may target cycles
**1 … M** instead of only cycle 1.

### 4.1 Definitions
- `P_rev` = the previous structure's (sid−1) **last BOS zone max-expanded OUTER** —
  the SAME value `_get_prev_bos_outer` (`orchestrator.py:177`) computes today for the
  Scenario-1 revert check. (Single source of truth — reuse it, do not re-derive.)
- `M` = the **earliest** cycle of the new structure whose **CTS extreme** clears
  `P_rev` (past it in the new structure's sd direction: sd=+1 ⇒ `CTS_M.price >= P_rev`;
  sd=−1 ⇒ `<= P_rev`). The ceiling check at target `T` only inspects cycles
  `[0, T−1]`, all of which are completed → use each prior cycle's **locked/confirmed
  CTS extreme** (`_cts_by_cycle`), not a still-updating running extreme.

### 4.2 Rule
- The post-reversal cross target cycle ranges **1 … M**. Cycle `M` gets the
  final/largest cross; cycles **> M** fall back to normal single fibs.
- **Start anchor is the UNCHANGED dead-cycle walk** (`earliest_x` can be 0, 1, …).
  `M` is **purely a ceiling on the target**, never a floor on the start and never a
  reason to force a cross.
- Operationally: **allow a cross at target T iff no cycle in [0, T−1] has yet cleared
  `P_rev`** — the cap fires the instant a CTS clears the old BOS outer.

**Mental model (avoid the single-object trap):** "a cross from 0 to M" is NOT one
ever-growing object spanning 0→M. As in Path B, a cross is **recomputed at each target
cycle** 1…M — each spans `earliest_x→target` with its own backward walk; the previous
target's cross is obsoleted (new-cycle terminal) when the next forms. What a viewer
sees is the *currently active* cross's **target advancing** cycle-by-cycle while its
**start stays at the earliest live cycle** (typically 0), until target reaches `M`;
target `M+1` forms no cross (normal single fib). The integer-version axis (§8) tracks
**start-anchor** (`earliest_x`) shifts within a target, NOT the target advance.

### 4.3 Within the cap, the requirements still gate (do NOT bypass)
For a target T ≤ M, the cross still forms ONLY if the §3 dead-cycle walk + own-imbalance
conditions hold (exactly Path B). `M` adds a ceiling; it never manufactures a cross
the imbalance logic wouldn't otherwise produce.

### 4.4 Edge cases / invariants
- **`M = 0`** (CTS_0 already clears `P_rev`): no room for a cross (a cross needs span
  ≥ 2 cycles) → cycle 0 stays single-only, no cross ever forms = today's behavior.
  **REVISIT AT IMPLEMENTATION** + check what the code currently does for an immediate
  clear (user-flagged 2026-06-16).
- **`M` undefined:** cannot happen for a structure that reverses — DOMAIN INVARIANT: a
  reversal REQUIRES some cycle's CTS to clear the threshold (that clearing is what
  makes the reversal possible), so another reversal can't occur before a cycle clears
  `P_rev`. For a still-open structure at the data edge, `M` is simply not reached yet →
  the cross keeps extending each new cycle until `M` arrives. No special case.
- **`sid == 0`** (no previous structure): `P_rev` undefined → no ceiling concept; sid=0
  retains "simple flow" (no cross at all) unless/until a future decision changes it.

---

## 5. Main-only — the Scenario-1 / revert outer gate (retained, INDEPENDENT)

Scenario 1 (CTS_0 ≥ reversal idx → cycle-0 legit post-reversal) + the revert (BOS_1
touches prev-BOS-outer → S1 FALSE, kill cycle-0 fib) are **kept as-is** and treated as
an **outer gate** wrapping the unified routine:
- S1 TRUE (post-revert) → normal single fibs (no cross), as today.
- S1 FALSE → hand to the unified cross routine (now ceiling-capped at `M`).

**Independence (user-confirmed 2026-06-16):** the revert and the ceiling both key off
the *same* `P_rev` value but with different anchors (revert tests `BOS_1` touching
*into* the zone; the ceiling tests `CTS_x` clearing *past* it) — they stay fully
independent. Revert still kills the cycle-0 fib + routes to S2/S3 exactly as today; the
ceiling only governs how far the resulting cross extends. (May revisit a merge later.)

---

## 6. What main does NOT adopt — the pre-established phase

Path B's `pre_established` cross (created during cycle n's tail on
`CTS_THRESHOLD_UPDATED`, anchored on the prospective BOS_{n+1}) is **subordinate-only
and NOT adopted on main**. Main evaluates the cross only at CTS-established/updated
events (as Path A does today). The shared routine's pre-established ingredient is gated
OFF for `fib_mode == "h1"`. (This is also why main fibs have no §6-lifecycle-spec
early start — see `FIB_LIFECYCLE_SPEC §6`.)

---

## 7. Subordinate (Path B) retained behavior

Subs keep: generalized unbounded target (no `M` ceiling), the pre-established phase,
integer versioning in `_fibs`, dead-cycle cache, cycle-0 single + late-activate. The
unification must leave sub output **byte-identical** (the refactor extracts the routine
subs already use; subs are the reference implementation).

---

## 8. Storage & versioning unification depth — DECIDED: full unification (user 2026-06-16)

**DECISION (user 2026-06-16): Option (1) — FULL storage unification.** Main adopts
Path B's integer-versioned `_fibs` storage (`(sid,cyc,"cross",v)` + single fallback)
and **retires `_cross_cycle_data`'s named slots.** One storage model; versioning +
lifecycle logging genuinely uniform; main gets multi-version crosses — which the
multi-cycle ceiling (§4) needs anyway (a 0→M cross shrinks/extends across versions
like a sub's). This **supersedes `FIB_LIFECYCLE_SPEC §5`'s "project, don't unify"**
(which §5 itself flagged as "not closed — may revisit"); this arc IS that revisit.

**Consequence — accepted:** it refactors main-structure fib SELECTION, the
byte-identical-risk zone §5 avoided. Managed by staging it as a **byte-identical
extract + reroute first** (§11a — the named-slot ↔ `v0` mapping is 1:1 at `target=1`),
then the behavioral multi-cycle extension (§11b), each its own `/compare` + sign-off.

The rejected Option (2) (keep named slots, share decision-only) was declined because
named slots (≤2 candidates) cannot represent a 0→M multi-version cross, so it would
have had to grow into Option (1) to support §4 regardless.

---

## 9. Proposed shared routine (signature sketch — full unification per §8)

A pure function (new module `zones/cross_cycle_fib.py`, or extend
`select_fib_anchor_for_cycle`), taking the per-cycle geometry + imbalance df + the
mode parameters, returning the cross decision. The `_dead_cycles` cache and the
`_bos_by_cycle`/`_cts_by_cycle` maps are FibTracker instance state → passed in as
arguments to keep the routine pure (mirrors how `select_fib_anchor_for_cycle` takes
`c0_data` rather than reading tracker state).

```
resolve_cross_cycle_anchor(
    df, sid, target_cycle, sd,
    bos_by_cycle, cts_by_cycle,        # per-cycle anchors (caller-supplied maps)
    dead_cycles,                       # mutable cache set (walk updates it)
    own_imb_start, current_candle,
    anchor_idx, anchor_price,          # CTS-side running extreme
    fill_threshold,
    target_ceiling=None,               # main: M; sub: None  (§4)
    fill_as_of="current",              # sub: "current"; main: per-cycle snapshot policy (§3.1)
) -> CrossDecision   # {earliest_x, action ∈ {none, create_v0, extend, shrink, fail},
                     #  anchor geometry, version}
```

`_m15_cross_check` becomes a thin wrapper that calls this + applies the version
transition to `_fibs`. Path A's `_handle_cycle1_scenarios` S1-FALSE branch calls the
same, passing `target_ceiling=M`. (Exact shape to be finalized with the §8 decision.
As landed, §11a-i: `resolve_cross_cycle_eligibility`, which since Plan F also takes a
keyword-only, required `evaluated_at` — §3 item 1.)

---

## 10. Edge cases & invariants (consolidated)

1. Imbalance/dead-cycle conditions still gate within the ceiling (§4.3).
2. `M` always defined for a reversing structure; not-yet-reached for an open one (§4.4).
3. `M = 0` → no cross (§4.4) — **revisit at implementation + check current behavior.**
4. Scenario-1/revert independent of the ceiling (§5).
5. Sub output byte-identical (§7).
6. Main `target=1` reproduces today's S2/S3 byte-identically (§3 sanity floor).
7. **In-flight POI resolver agreement (`market_structure.py:1928`).** The MS in-flight
   resolver shares `select_fib_anchor_for_cycle`, which is **cycle-1-only** today. For
   §11a (byte-identical, `target=1`, snapshot fill-as-of) agreement is automatic
   (since Plan F on cond2 / cond3 only — cond1 is cut at the CTS_1 moment downstream,
   not in-flight: the accepted M1 divergence, §2.1). But
   **§11b is bigger than "pass M":** extending main to target ≥ 2 means the in-flight
   resolver needs the **whole generalized walk** (with the snapshot fill-as-of), not
   just the `M` value — else the in-flight POI snapshot and the downstream fib diverge
   for cycles ≥ 2. Treat the in-flight resolver generalization as a first-class 11b
   sub-task, not an afterthought.

---

## 11. Implementation staging

Two steps, each its own `/compare` + sign-off (project norm: one step per session,
preserve config windows, show replay timing).

### 11a — Extract + reroute, BYTE-IDENTICAL
Split into two sub-commits, each its own byte-identical `/compare`:

**11a-i — decision extract (DONE 2026-06-17, `d6b8f54`).** Lift §3's cross-requirements
into the shared routine `zones/cross_cycle_fib.py::resolve_cross_cycle_eligibility`;
reroute BOTH `_m15_cross_check` (sub, `fill_as_of=current`) AND — via the thin
`select_fib_anchor_for_cycle` wrapper — Path A's S1-FALSE branch (main, `target=1`,
`fill_as_of=snapshot`) + the in-flight POI resolver through it. No behavior change. The
§3.1 fill-as-of parameterization is what *makes* main byte-identical (without it, B's
current-semantics walk would drop main's superset crosses). Storage untouched.

**11a-ii — main storage migration, RELOCATE-ONLY (DONE 2026-06-17).** Main's Scenario-2
cross relocates from the `_cross_cycle_data` named slot + `_fibs[(sid,1)]` mirror into the
versioned key `(sid,1,"cross",0)` + `_cross_version` (the `cycle0` decision-input dict
stays; `normal_cycle1`/`cross_cycle` slots retired). `normal_cycle1` is no longer built
upfront — **create-on-fail** materializes a single at `(sid,1)` only when the cross fails.
**Decided relocate-only (not full-adopt):** the fib_lifecycle CSV derives `version`/
`fib_mode` from *meta* and exports `meta` verbatim, so adopting the sub's meta would
change those columns → NOT byte-identical. So main keeps its bespoke meta (no
`version`/`fib_mode` keys) under a versioned key; the cosmetic meta-unification is
**deferred to 11b** (where main rows change anyway). Byte-identical on this window because
the sole main cross (sid=1 cyc=1) wins throughout → one record, key-relocated, meta + the
`new_cycle@902` terminal + the phantom collapse all preserved. NB the h1 FALLBACK-to-single
path is **unreachable** (cond1/cond3 fixed across cycle-1 updates; cond2 == the normal's
own check) — create-on-fail is dead/harmless for h1, exercised only by 11b multi-cycle
crosses. **Landmine learned:** the shared `_activate_fib`'s new versioned-cross obsolete
must be gated to `fib_mode=="h1"` — ungated it flipped a sub cross's `end_reason`
`next_cycle→new_cycle` (subs obsolete their crosses via `_m15_create_cross`).

### 11b Part A — Bounded multi-cycle cross capability on main (DONE 2026-06-17, byte-identical)
Wire `P_rev` (from `_get_prev_bos_outer`, now passed at EVERY sid≥1 cycle) → the
target ceiling `M` into FibTracker; route main cycles **1…M** through the shared cross
machinery, cycles **>M** to plain single. Implemented:
- `_prev_bos_outer[sid]` stash (set-once) + `_cross_allowed_for_target(sid, T, sd)` =
  "no cycle in [0, T−1] cleared P_rev" (= `T ≤ M`). `M=0` (CTS_0 clears) → no cross.
  P_rev absent → no cross for cycles ≥2 (conservative); cycle-1 cross only suppressed on
  a **definite** M=0 (`_prev_bos_outer` present), preserving pre-11b behavior otherwise.
- Cycle ≥2 EST: a read-only eligibility **peek** (dead-cycle COPY) decides cross-vs-single
  so a no-cross outcome is byte-identical; an actual cross runs `_m15_cross_check`
  (`fill_as_of="current"`). Cycle ≥2 UPD maintains an existing cross; CONF lock
  generalized to any cycle with a versioned cross. Cross is born at EST only (main keeps
  its cycle≥2 one-shot semantics).
- **`fill_as_of` decided: "current"** for the deeper (target ≥2) walk (no legacy A
  behavior; matches subs). Cycle-1 keeps "snapshot" (the 11a-i wrapper) for continuity.

`/compare` byte-identical (21/21) vs `20260616_185639_d1bd7bb` — see §12.1: M=2 *allows*
a cycle-2 cross but cycle 1 is dead by cycle-2 time, so none forms (§4.3 — the imbalance
walk still gates; M never manufactures a cross). The capability is correct + unit-tested
(`test_main_versioned_cross.py`: the cross-forming path is proven on synthetic data) but
**dormant on this window**. Mirrors true-first-breakout Commit 2 (byte-identical,
capability live but output unchanged).

### 11b Part B — in-flight POI resolver generalization (§10.7) — DEFERRED 2026-06-17
The MS in-flight resolver (`compute_poi_inners_for_cycle`, cycle-1-only) must generalize
to the full walk so MS's proximity-based CTS confirmation agrees with the downstream
cross for cycles ≥2. **Deferred because no main cycle≥2 cross forms on the current
window** → there is nothing for `/compare` to exercise/validate, and the change is a
substantial MS-protocol extension (MS would need per-cycle BOS/CTS maps + `P_rev`).
**Bounded gap:** it only bites when a main cycle≥2 cross actually forms AND that cycle's
CTS confirms via POI proximity — neither happens here. Revisit when a window (e.g. via
incremental config-window expansion) produces a main multi-cycle cross; then implement +
`/compare`-validate Parts A and B together.

---

## 12. Validation

- Unit tests for the shared routine: single-step (= A's cond1/2/3) vs multi-cycle (= B's
  walk); ceiling `M` (target capped, `M=0`, not-yet-reached); imbalance-still-gates.
- `test_cross_cycle_fib.py` must stay green or be updated deliberately.
- `/compare`: 11a byte-identical (all 21 CSVs); 11b main POI/fib shifts only where a
  multi-cycle cross forms, subs unchanged. Pause for chart review (post-reversal cross
  is hard to eyeball — verify on the H1 chart).
- Fib state is mostly latent (no fib CSV; M15 fib lines off) — the surfacing path is
  **POI** (`FIB_LIFECYCLE_SPEC §15.8`). For 11b, validate via POI CSVs/charts AND a
  temporary fib-record dump (`debug/export_fib_lifecycle.py`).

### 12.1 Worked validation case — the `M = 2` anchor (11b ground truth)

Verified against baseline `20260616_185639_d1bd7bb` (config window
`2025-11-15 → 2026-01-20`, NZD_USD H1). This is the **concrete expected outcome
that proves the §4 ceiling fired** — capture it before 11b so the next session
has a target.

- **Reversal:** sid=0 (bullish, sd=+1) reverses at idx 897 / apply 902 →
  new sid=1 (bearish, sd=−1). `bos_frozen = 0.5736`.
- **Threshold coincidence (checked):** `_get_prev_bos_outer(sid=1) = 0.5736`
  too — sid=0's last BOS zone (cycle 1, `buy`, outer 0.5736) has a single INIT
  `bounds_steps` (no expansion), so max-expanded outer = bottom = `bos_frozen`.
  So `P_rev` (the §4 ceiling) and the reversal `bos_frozen` are the **same
  value** here; no drift.
- **sid=1 CTS extremes** (clear ⇔ `CTS ≤ 0.5736` for sd=−1):

  | cycle | CTS extreme | vs 0.5736 | clears? |
  |---|---|---|---|
  | 0 | 0.57902 | above | no |
  | 1 | 0.574 | above (~4 pips) | no |
  | 2 | 0.57112 | **below** | **yes** |

  → **M = 2** (earliest cycle whose CTS clears `P_rev`). Note CTS_1 sits only
  ~4 pips above the line — the boundary is close, so confirm the clear
  comparator matches the revert check's inclusive "touch" (`≤`/`≥`) + the same
  EPS the codebase uses (not a boundary case in *this* data, but tight).

- **Ceiling result: M = 2** → the §4 ceiling *allows* a cross to target cycle 2.

**ACTUAL 11b-Part-A OUTCOME (2026-06-17): no cycle-2 cross forms → byte-identical.**
The earlier "cycle 2 gets the final cross" expectation was OPTIMISTIC — it conflated
"M allows cycle 2" with "the imbalance walk produces a cross at cycle 2." It does not,
because the dead-cycle walk still gates (§4.3). Measured unfilled sd-imbalance as of
cycle-2 time (idx 902):

  | cycle | range | unfilled sd-imbalance @902 |
  |---|---|---|
  | 0 | [689,710] | True (live) |
  | 1 | [728,761] | **False (DEAD)** |
  | 2 own | [826,902] | True |

The walk from target=2 hits cycle 1 first → **dead → stops** (contiguity: a dead cycle 1
blocks reaching the live cycle 0) → no eligible prior → **no cross**. So sid=1 cycle 2
stays a plain single exactly as before. `/compare` byte-identical (21/21). The lesson:
**`M` is a ceiling, not a producer** — a real main multi-cycle cross needs `M ≥ 2` AND a
*contiguous* live run of prior cycles, which this window doesn't have. To exercise an
actual main multi-cycle cross (and Part B), expand the config window to find one.
  (`M = 0` — CTS_0 already clears — is NOT exercised by this window; define +
  unit-test it separately in 11b, no real-data anchor here.)

---

## 13. Open items / deferred

- ~~§8 storage depth~~ — **DECIDED 2026-06-16: full versioned `_fibs` unification.**
- ~~§3.1 fill-as-of (11a)~~ — **DECIDED 2026-06-16: parameterize (snapshot-preserving),
  11a byte-identical.**
- **Fill-as-of convergence (main → "current" like sub)** — DEFERRED open question, its
  own intentional-diff step if ever taken (§3.1); not decided, not in scope now.
- **Deeper-walk fill-as-of (11b)** — A defines snapshot only for target=1; the
  intermediate cycles in a target ≥ 2 walk need a decision (recommend "current"; §11b).
- **`M=0` behavior** — confirm + check current code at implementation (§4.4).
- **Scenario-1/revert ↔ ceiling merge** — kept independent now; may revisit (§5).
- **`sid==0` cross** — out of scope; simple-flow retained (§4.4).
- **In-flight POI resolver generalization** (§10.7) — 11b must generalize the MS-side
  `select_fib_anchor_for_cycle` caller to the full walk (not just `M`), else cycles ≥ 2
  diverge in-flight vs downstream.
- **Single-cycle fibs coexisting with cross** (`project_sub_single_cycle_fib_pois.md`)
  — separate, additive, still deferred; not part of this unification.

---

## 14. Decision log

1. Unify the cross-cycle **decision logic** (requirements + walk + versioning) into one
   shared pure routine; mode differences are parameters, not separate paths.
2. Main gains a **bounded multi-cycle** cross (target 1…`M`); subs stay unbounded.
3. `M` = earliest cycle whose **CTS** clears `P_rev` (= prev-BOS-outer, reuse
   `_get_prev_bos_outer`); a **ceiling on the target**, never a start floor, never
   forces a cross.
4. Main does **NOT** adopt the pre-established (`CTS_THRESHOLD_UPDATED`) phase.
5. Scenario-1/revert kept as an **independent outer gate** on main.
6. Dead-cycle walk + imbalance requirements **unchanged**; still gate within the ceiling.
7. Sub output stays byte-identical; main `target=1` reproduces today's S2/S3.
8. **DECIDED (2026-06-16):** storage depth = **full versioned `_fibs` unification** —
   main retires named slots, supersedes FIB_LIFECYCLE §5. (Option (2) decision-only
   rejected: named slots can't hold a multi-version 0→M cross.)
9. Staging: (a) extract + reroute byte-identical, (b) bounded extension on main.
10. **DECIDED (2026-06-16):** A and B differ in cycle-0 fill-as-of timing (A snapshot
    @CTS_0/BOS_1, B @current; A is a superset). **Parameterize `fill_as_of`** to
    preserve both → 11a byte-identical. Converging main onto "current" is a deferred,
    separate intentional-diff step, NOT part of the refactor.
