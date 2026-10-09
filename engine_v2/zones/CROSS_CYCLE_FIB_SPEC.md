# Cross-Cycle Fib — ONE logic for every structure

> **Status: CANONICAL since PLAN_I S4 (2026-10-08).** Every structure — an H1 sid or a unique sub (its sid 0) — runs
> the same FibTracker logic (`zones/fib_tracker.py`); the H1-main logic it replaced (Scenario 1 / 2 / 3, the revert, the
> ceiling `M` on the previous BOS zone's outer `P_rev`, main's snapshot fill-as-of, the `fib_mode` parameter) is kept
> VERBATIM in **Appendix A** — the last commit that ran it is `0846441` (PLAN_I S3b). The rules are the user's
> (`plans/PLAN_I_cross_cycle_fib_revamp.md` §1.2 + §1.4; decisions §6, verbatim in `plans/plan_i_inputs/
> cross_cycle_fib_spec.md`); the build record is PLAN_I §9.10.
>
> **Companion spec:** `FIB_LIFECYCLE_SPEC.md` (the lifecycle axes: start / end / status, Option A, the floor clamp).
> This spec is about WHICH fib a cycle gets and its anchors.
>
> **MarketStructure reads this same decision since PLAN_I S5 (2026-10-08):** an embedded FibTracker (§8); MS's retired
> Scenario-2 in-flight read is in **Appendix A, part 2** (verbatim; the last commit that ran it is `0de98ca`, PLAN_I S4).
>
> **Still to come (PLAN_I §9.4):** S5b — a cycle's lifecycle end caps its fib's end (PLAN_I R9-3, F19); S6 — the
> `fib_mode` meta key / CSV column is retired.

---

## 0. One-sentence summary

Cycle 0 gets a single fib; a cycle T ≥ 1 gets a CROSS fib back across its live, contiguous prior cycles when the
structure's cross-fib LIMIT allows it (a reference price `P` fixed by the structure's starting trigger, and no earlier
cycle's CTS anchor strictly beyond `P`), else a single — main and sub alike.

---

## 1. Where it runs

- **One tracker per downstream run:** `orchestrator._run_downstream_pipeline` builds `FibTracker(FibTrackerConfig(…),
  cross_fib_limits=…)` — the H1 main with `compute_main_cross_fib_limits(events)` (`zones/structure_lifecycle.py`), a
  sub's projection (`pooled_structure_build.project_to_window`) with `{0: sub.cross_fib_limit}`.
- **Dispatch order:** events sorted by `event_fields.processing_order_key` (CTS_ESTABLISHED / BOS_CONFIRMED / a
  pattern-path CTS_UPDATED at their ANCHOR; LANDMINES "Event Sort Order Is a Dispatch Invariant"). Handlers:
  `on_cts_established(event, df, bos_idx, bos_price)`, `on_cts_updated(event, df)`, `on_cts_confirmed(event)`,
  `on_cts_threshold_updated(event, df)`; then `set_reversal_terminals` and `_finalize_lifecycle_fields`.
- **Three order tripwires (`FibEventOrderError`, a `RuntimeError`):** the tracker raises when cycle T's
  CTS_ESTABLISHED arrives before cycle T−1's CTS_CONFIRMED (PLAN_I F14, S2); the dispatch raises when a CTS_ESTABLISHED
  arrives before its own cycle's BOS_CONFIRMED (F17, S2b — it used to be dropped silently); and MarketStructure raises
  at the end of every run whose embedded tracker (§8) saw an EMISSION order other than `processing_order_key` order
  (`check_processing_order`, S5 — the F16 and F17 (i) shapes; F17 (ii) trips only the dispatch). 0 on the reference
  window; every shape comes from MS (PLAN_I §9.2 F14 / F16 / F17) and stays loud until the MS fix planned after S5
  (PLAN_I R9-1 / R9-2).
- **Every TIME half is the handled event's MOMENT** (`ef.event_moment`; Plan E E3a): activation / deactivation /
  reactivation stamps, fill horizons, the knowability cut (`evaluated_at`, Plan F — `IMBALANCE_FILL_SEMANTICS.md`).

---

## 2. The rule per cycle

| Cycle | Limit (`_cross_allowed`, §3) | Fib |
|---|---|---|
| 0 | never consulted | **single** (§5) |
| T ≥ 1 | allowed | the **cross machinery** (§4) — a cross `BOS_x → CTS_T` over the live run, else the `cross_failed` single |
| T ≥ 1 | not allowed (no `P`, or a cycle k < T beyond `P`) | **single** (§5), `via: "cross_fib_limit"`; no pre-established cross |

**I8 — a single and a cross are never both ACTIVE on one cycle** (the user's "no coexistence"): asserted at every single
activation / reactivation (`_assert_no_active_cross`) and every cross creation / revival (`_assert_no_active_single`).
A superseded (inactive) record of the other kind may remain — the `cross_failed` fallback deactivates the cross first.

**A fib that starts after its cycle's establishment** (a single activated late on a CTS_UPDATED, or a cross created
late) ends the previous fib at that establishment, not at the late start (`_prev_cycle_end_at`, PLAN_I S3b F15 /
F18); the pre-established cross (created before the establishment) ends it at its creation (Option A,
`FIB_LIFECYCLE_SPEC.md` §7). "The previous fib" is the structure's LAST cycle that has one (`_current_cycle`): when a
cycle in between got no fib, the earlier fib ends at this later establishment, not at its own cycle's end (the stream
terminal is set-if-absent, so it beats finalize's pass-through end) — PLAN_I §9.2 F19, latent (0 on the reference
window; one real instance in 2,400 exact real slices). Decided (R9-3): the cycle's lifecycle end will CAP the fib's
end — stage S5b.

---

## 3. The cross-fib LIMIT

- **`P` (`cross_fib_limit_price`) — fixed once, at the structure's start, by its STARTING trigger** (PLAN_I §1.2, I1:
  the variation attaches to the trigger, not to the sid number). `CrossFibLimit(price, source, from_pool_idx /
  from_sub_id, started_by)` (`zones/cross_cycle_fib.py`):

  | Started by | `P` | `cross_fib_limit_source` |
  |---|---|---|
  | main, start trading (sid 0) | none | `none` |
  | main, a reversal (sid N) | the realised reversal's `bos_frozen` (`STATE_CHANGED(to=reversal)`, == the open watch's frozen barrier) | `reversal_bos_frozen` |
  | sub, reversal | the reversing sub's reversal `bos_frozen` | `reversal_bos_frozen` |
  | sub, first_confluence on parent cycle 0 | none | `none` |
  | sub, first_confluence on parent cycle c ≥ 1 | the parent's CTS_ESTABLISHED(c) `cts_frozen` | `parent_cts_threshold` |
  | sub, first_counter / subsequent_* | the previous sub x−1 at the record start: same direction → its confirmed `cts_threshold`, else its latest CTS anchor; opposite → its BOS threshold, else its starting extreme; no x−1 → none | `prev_sub_cts_threshold` / `prev_sub_latest_cts` / `prev_sub_bos_threshold` / `prev_sub_starting_price` / `none`; a same-direction x−1 with no CTS → `prev_sub_no_cts` (no `P`, a WARNING) — the set: `cross_cycle_fib.CROSS_FIB_LIMIT_SOURCES` |

  The sub formula: `multitf/cross_fib_limit.py` (`cross_fib_limit_for`), computed at the sub's first trigger's fire
  and re-derived at its start by the R6-2 tripwire (`check_cross_fib_limits`, warn-only). The values on the reference
  window: PLAN_I §9.5 a.
- **`beyond(p, P, sd)`** — STRICT: sd = +1 `p > P`, sd = −1 `p < P`; reaching `P` is not beyond.
- **`M` (`cross_fib_limit_cycle`)** = the first cycle whose LATEST CTS anchor (`_latest_anchor`: written at its
  CTS_ESTABLISHED, every CTS_UPDATED until its confirmation, and its CTS_CONFIRMED — the confirmed anchor wins) is
  beyond `P` = the LAST cycle a cross may target. `_cross_allowed(sid, T)` ⟺ `P` set and (`M` is None or T ≤ M). The
  cycles k < T are confirmed before T exists (the S2 tripwire) and a confirmed anchor never moves, so the answer is
  fixed for T's whole life.
- **REQUIRED:** a tracker consulted for a cycle ≥ 1 of a structure with no limit entry raises `CrossFibLimitError` (a
  `RuntimeError`, never swallowed by a sub geometry's `except (ValueError, IndexError)`).
- **Exported:** every fib record's meta carries `cross_fib_limit_price` / `_source` / `_cycle` (the structure's FINAL
  `M`) / `_from_sub_id` (+ main's `started_by`), stamped at finalize; the fib hover's last line
  (`_zone_render.fib_hover_limit_text`, CHARTING_SPEC).

---

## 4. The cross machinery (a cycle T ≥ 1 within the limit)

- **Phase machine** `_phase[(sid, cycle)] ∈ {pre_established, established, confirmed}`: `pre_established` from
  CTS_{T−1} CONFIRMED, `established` at CTS_T ESTABLISHED, `confirmed` at CTS_T CONFIRMED. Cycle 0 has none.
- **Pre-established cross** (on CTS_THRESHOLD_UPDATED of cycle T−1, target T in `pre_established` and within the
  limit): anchor = the running extreme past CTS_{T−1}'s confirmation (`_running_extreme_anchor`), own-imbalance start =
  the prospective BOS_T (`_find_prospective_bos`, the deepest pullback since then). Main has it too since S4 (PLAN_I
  I2; on the reference window main's (1,1) cross is created at 741 instead of 748).
- **The cross check** (`_cross_check`, at CTS_ESTABLISHED, every CTS_UPDATED and the threshold path):
  (1) the target's OWN window — `[BOS_T, CTS anchor]` at the establishment / an update, `[prospective BOS, the
  processing candle]` on the threshold path — must hold an unfilled sd imbalance formed by the moment — else the active cross deactivates (`own_imb_filled`) and the cycle's existing single re-checks its own
  window (S3b F5: it extends and deactivates); (2) the **dead-cycle walk** from T−1 down to 0
  (`resolve_cross_cycle_eligibility`): cycle k is dead when `[BOS_k, CTS_k]` holds no unfilled
  sd imbalance filled to the current candle (cached in `_dead_cycles[sid]`, dead stays dead); the walk stops at the
  first dead cycle → `earliest_x`; (3) the version transition, keyed by the START anchor: no version → create v0;
  same start → extend the CTS anchor in place; start moved up (`cross_shortened`) → v+1; an inactive version with the
  same start → revive in place, with a moved start → a new version v+1; `earliest_x == T` (no eligible prior) → `cross_failed`: deactivate the cross and, in
  the established phase, activate the single (`via: "cross_failed"`).
- **CTS_CONFIRMED** locks the active cross and the active single of the cycle and moves T+1 to `pre_established`.

---

## 5. The single

`_activate_fib` stores it at `(sid, cycle)`: activated at the cycle's CTS_ESTABLISHED when its own `[BOS, CTS]` holds
an unfilled sd imbalance formed by the moment, else at the first CTS_UPDATED that finds one (`activated_on:
"update"`); every update re-checks it (`_update_fib_cts`: extend, then deactivate when all its imbalances are filled /
reactivate when one appears); CTS_CONFIRMED locks it. Labels (`meta["via"]`): none on cycle 0; `cross_fib_limit` past
the limit; `cross_failed` for the fallback of an allowed cycle.

---

## 6. Storage and meta

- Keys in `_fibs`: `(sid, cycle)` the single; `(sid, cycle, "cross", v)` the cross versions (`_cross_version`).
- Meta = the sub path's for every structure (Q8): a cross carries `fib_mode` / `cross_cycle` / `cross_start_cycle` /
  `version` / `activated_at` (+ `locked_at` / `deactivated_*` / `reactivated_at`); a single `activated_at` / `fib_mode`
  (+ `via`, `activated_on`, …); `obsolete_reason: "new_cycle"` on a record a next cycle ended; the cross-fib limit keys
  (§3). `fib_mode` is always `"cross_cycle"` until S6 retires it. Retired with main's old logic (a rule-3 migration,
  S4): `cycle1_bos_anchor_idx` (also out of `entity_df_mutation._FIB_META_IDX_KEYS`), `scenario`, `flow`, the
  `scenario1_revert` end reason (`_INVALIDATION_END_REASONS` is empty — the `disappeared` status stays as a hook).
- One fill constant: `fill_threshold = 0.70` (`FibTrackerConfig`).

---

## 7. The shared routine

`zones/cross_cycle_fib.py` `resolve_cross_cycle_eligibility` — pure: the own test + the dead-cycle walk → `earliest_x`
/ `crosses` (+ the resolved anchors), each prior cycle checked to the current candle; FibTracker is its only caller
(`fill_horizon_idx` = `evaluated_at` = the handled event's moment). The retired Scenario-2 selector's "snapshot"
fill-as-of policy (`fill_as_of` / `prior_cached_liveness` / `snapshot_horizon_idx`) and the reserved `target_ceiling`
were deleted in S5 with MS's resolver, their last reader (Appendix A, part 2).

---

## 8. MarketStructure's in-flight read — the same decision (PLAN_I S5, D5)

MS's sd-zone-proximity CTS confirmation (`_maybe_confirm_cts_via_proximity`) reads a POI inner refreshed at every
CTS_ESTABLISHED / CTS_UPDATED. Since S5 that inner comes from THIS logic:

- **The embedded tracker.** `structure_engine._make_market_structure` builds a `FibTracker` — the downstream's config
  (`DEFAULT_FIB_LEVELS`, `FIB_FILL_THRESHOLD` = 0.70), `quiet=True` (no `[fib]` lines from every geometry / probe MS),
  the run's REQUIRED `cross_fib_limit` keyed by its `structure_id` — and injects it with `poi_zones.poi_inner_for_fib`
  (MS imports nothing from `zones/`). The limit: main sid N ← `compute_main_cross_fib_limit(events of sids < N, N)`
  (the helper `compute_main_cross_fib_limits` maps for the downstream — equal by construction); a sub's geometry ←
  the pool's `P` (§3, computed at the fire; `_build_geometry` → `compute_bounded_structure`); the probe's Phase 2 ←
  `CrossFibLimit(None, "none")` explicitly (PLAN_I F7: its result never depends on the read).
- **Fed at emit.** MS hands the tracker each CTS event as it is emitted: CTS_ESTABLISHED after its cycle's
  BOS_CONFIRMED (with `st.bos`, the pairing the downstream dispatch makes), every CTS_UPDATED (raw and pattern path),
  CTS_CONFIRMED, CTS_THRESHOLD_UPDATED (`MarketStructure._feed_fib_tracker`). The tracker's own `RuntimeError`s pass
  through; any other error is re-raised as `EmbeddedFibTrackerError` (a `RuntimeError` — the sub geometry's `except
  (ValueError, IndexError)` cannot drop the sub).
- **The read.** At each refresh: `FibTracker.representative_fib(sid, cycle)` — the cycle's ACTIVE latest cross, else
  its ACTIVE single, else None — handed to `poi_inner_for_fib(df, fib, fill_threshold)`, which reads the frame's
  `attrs["timeframe"]` first on EVERY call (a no-fib refresh included; PLAN_H PR1, PLAN_I F11) and returns the
  institutional-pattern pick's inner (`select_ip_ic`, the drawn POI's selector), or None → the BOS inner only. Every
  imbalance question of the fib decision is asked at the fed event's moment (Plan F), so the M1 divergence and the
  fib the tracker never creates / has deactivated are gone (PLAN_I §9.2 F9).
- **The order.** The drawn fib is decided in `processing_order_key` order, MS's in EMISSION order; the two are the same
  sequence unless a fib-handled event of a structure has its stamped idx in one of its CTS_ESTABLISHED's
  [anchor, moment) (F14; F17 (i)) or a CTS is confirmed with an anchor AT OR after its confirmation candle (F16 — at
  an equal idx `processing_order_key` sorts the confirmation first). F17 (ii) — a cycle whose BOS anchor follows its
  CTS anchor — keeps the order (the establishment and its BOS share their moment): MS completes, only the downstream
  dispatch raises. At the end of every MS run `check_processing_order()` compares them and raises
  `FibEventOrderError` on a difference (the run is complete when it raises — `ms.events` / `ms.df` are whole) — in
  the main run, a sub geometry AND the probe's Phase 2 (kept, S5 landing review m1: the probe's result never depends
  on the read, F7, but an F16 / F17 (i) shape inside a probe iteration stops the replay too, even for a start never
  built). 0 on the reference window and on 400 random real slices (lens A); PLAN_I R9-1 / R9-2: the MS fix comes
  after S5 as its own plan.
- **Measured (PLAN_I §9.10, S5 step A, `plan_i_inputs/s5_agreement.py`):** the embedded tracker's handled sequence ==
  the downstream's (H1 sid 0 52 / 52, sid 1 24 / 24; each of the 8 subs' downstream sequence an exact prefix of its
  geometry's); the inner differs from the retired resolver's at 20 / 371 refreshes; 0 / 475 proximity answers change.

---

## 9. Open items

- **The MS fix (PLAN_I R9-1 / R9-2, after S5 — its own plan):** F16 — MS can confirm a CTS with an anchor after its
  confirmation candle (a range back-fill look-ahead; the drawn fib locks at the earlier anchor; MS's end-of-run
  tripwire now raises on it); F17's MS side — a cycle established on the previous cycle's CTS candle / a BOS anchor
  after its CTS anchor (a breakout establishing cycle T must break T−1's confirmed `cts_threshold` at its apply; the
  BOS window ends at the CTS anchor). Register `IDEA_PARKING_LOT.md` §D.1 item 2.
- **S5b (PLAN_I R9-3):** F19 — the cycle's lifecycle end caps its fib's end (§2).
- The old §13 items, reconciled: main's fill-as-of converged on "current" (S4); the deeper-walk fill-as-of is
  "current"; `M = 0` → no cross (pinned, S3); the Scenario-1 / revert ↔ ceiling merge is moot (both retired); `sid 0`
  has no `P` → singles, cycle 0 included (Q6); the in-flight resolver generalization and the sub snapshot's intra fib
  are resolved by S5 (§8); single + cross coexistence is ruled out (I8).

---

## 10. Decision log

The decisions are PLAN_I §6 (Q1–Q8, R0 … R8-3); the ones that shaped this spec: R0 / R0a (the limit, set by the
starting trigger), I1 (attach to triggers), I2 (main gets the pre-established cross), I5 (main's post-reversal handling
= the reversal-started sub's), I8 (no coexistence), I-lean (unify on the sub's logic), Q6 (cycle 0 single everywhere),
Q8 (the sub's meta format), R6-1 (d) + R7-3 (the past-limit single, labelled), R6-1b + R8-3 (S3b), R6-5 (S4 one stage),
R2-3 (this appendix), Q7 + R6-4 (S5: MS reads the same decision; the order tripwire raises), R9-1 / R9-2 (the MS fix
after S5), R9-3 (S5b). The 2026-06-16 design decisions of the replaced spec are in Appendix A ("14. Decision log").

---

## Appendix A — Retired logic (history)

> Moved VERBATIM from this spec as it stood at `0846441` (PLAN_I S3b, 2026-10-08) — the last commit whose FibTracker
> ran the H1-main logic below (PLAN_I R2-3). Section numbers are the old ones; their cross-references point into the
> old document (`git show 0846441:engine_v2/zones/CROSS_CYCLE_FIB_SPEC.md`). Nothing here describes the current code.

#### 2.1 Path A — H1 main, post-reversal (Scenario 1/2/3)

- **Scope:** `sid ≥ 1` only, and **only the cycle-0 → cycle-1 cross**. `sid == 0`
  never crosses ("simple flow"). Cycles 2+ get plain single fibs.
- **State:** `self._scenario1[sid] ∈ {None, True, False}`. Scenario 1 decided at
  CTS_0 EST/UPD (`_handle_cycle0_scenario1`): the CTS_0 event's MOMENT `>= reversal_confirmed_idx` (the EST `confirmed_at` / the
  update's moment since Plan E E3a; before it the CTS_0 anchor idx)
  → TRUE (cycle-0 single fib unlocked); resolves FALSE at CTS_0 CONFIRMED if never
  reached. Cycle-0 data cached as a plain dict in `_cross_cycle_data[sid]["cycle0"]`.
  Its `has_unfilled` is stored **uncut** by the c3 knowability rule: it is cond2,
  judged at its use (CTS_1 EST, when every gap in `[BOS_0, CTS_0]` has formed), which
  keeps it equal to the MS in-flight mirror `cycle0_data`. The Scenario-1 activation
  decided at CTS_0 EST/UPD asks at that event's moment instead (the cut check at EST,
  `_c0_has_unfilled_now` on updates) — Plan F 2026-09-24; `IMBALANCE_FILL_SEMANTICS.md`
  "Knowability — the c3 rule".
- **Cycle-1 decision** (`_handle_cycle1_scenarios`): if S1 TRUE → **revert check**
  (`_should_revert_scenario1` — BOS_1 reaches into prev structure's last BOS
  zone outer → flip S1 FALSE, kill cycle-0 fib via `_deactivate_cycle0_fib`, terminal
  `scenario1_revert`). If S1 (post-revert) TRUE → normal cycle-1 single fib. If S1
  FALSE → **Scenario 2 vs 3** via `select_fib_anchor_for_cycle` (3 static imbalance
  conds: cond1 cycle-1 unfilled, cond2 cycle-0 unfilled, cond3 BOS_1 hasn't filled
  cycle-0). All true → cross BOS_0→CTS_1 (`scenario_2_cross`); else single (Scenario 3).
- **Storage (pre-§11a-ii — today: §11a-ii):** named slots in `_cross_cycle_data[sid]` (`cycle0` dict,
  `normal_cycle1` FibState, `cross_cycle` FibState); only the active winner mirrors
  to `_fibs[(sid,1)]`. **≤2 anchor candidates** (BOS_0 cross vs BOS_1 normal), no
  integer version. Update toggles cross↔normal (`_update_cycle1_fibs`).
- **Shared util already used:** `select_fib_anchor_for_cycle` (`zones/fib_tracker.py`),
  a pure function ALSO called by the MS in-flight POI resolver
  (`MarketStructure._update_cycle0_data` / `_refresh_poi_inners_for_cycle`) so in-flight and downstream agree on Scenario 2. It
  covers **cycle 1 only** and only the S2-vs-S3 decision (not S1, not revert, not a
  walk). (Since Plan F, 2026-09-24, they agree on cond2 / cond3 but not always on
  cond1: FibTracker passes `evaluated_at` = the CTS_1 moment, the in-flight resolver
  `None` — the accepted M1 divergence, LANDMINES "Scenario 2 anchor agreement".)

#### 2.3 Divergence map

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

#### 3.1 The fill-as-of difference — A and B are NOT identical at `target=1` (decided 2026-06-16)

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

### 4. Main-only — the target-cycle CEILING `M`

This is the **new capability** (the "update" task): main's cross may target cycles
**1 … M** instead of only cycle 1.

#### 4.1 Definitions
- `P_rev` = the previous structure's (sid−1) **last BOS zone max-expanded OUTER** —
  the SAME value `_get_prev_bos_outer` (`orchestrator._run_downstream_pipeline`) computes today for the
  Scenario-1 revert check. (Single source of truth — reuse it, do not re-derive.)
- `M` = the **earliest** cycle of the new structure whose **CTS anchor** clears
  `P_rev` (past it in the new structure's sd direction: sd=+1 ⇒ `CTS_M.price >= P_rev`;
  sd=−1 ⇒ `<= P_rev`). The ceiling check at target `T` only inspects cycles
  `[0, T−1]`, all of which are completed → use each prior cycle's **locked/confirmed
  CTS anchor** (`_cts_by_cycle`), not a still-updating running extreme.

#### 4.2 Rule
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

#### 4.3 Within the cap, the requirements still gate (do NOT bypass)
For a target T ≤ M, the cross still forms ONLY if the §3 dead-cycle walk + own-imbalance
conditions hold (exactly Path B). `M` adds a ceiling; it never manufactures a cross
the imbalance logic wouldn't otherwise produce.

#### 4.4 Edge cases / invariants
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

### 5. Main-only — the Scenario-1 / revert outer gate (retained, INDEPENDENT)

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

### 6. What main does NOT adopt — the pre-established phase

Path B's `pre_established` cross (created during cycle n's tail on
`CTS_THRESHOLD_UPDATED`, anchored on the prospective BOS_{n+1}) is **subordinate-only
and NOT adopted on main**. Main evaluates the cross only at CTS-established/updated
events (as Path A does today). The shared routine's pre-established ingredient is gated
OFF for `fib_mode == "h1"`. (This is also why main fibs have no §6-lifecycle-spec
early start — see `FIB_LIFECYCLE_SPEC §6`.)

### 11. Implementation staging

Two steps, each its own `/compare` + sign-off (project norm: one step per session,
preserve config windows, show replay timing).

#### 11a — Extract + reroute, BYTE-IDENTICAL
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
crosses. **(2026-09-28: the h1 create-on-fail in `_update_cycle1_main` is DELETED — 0 hits
over the replay and the full suite, line-traced; the invariant it rested on is asserted
instead: a failed cross with its own `[BOS_1, CTS_1]` window still unfilled raises.)** **Landmine learned:** the shared `_activate_fib`'s new versioned-cross obsolete
must be gated to `fib_mode=="h1"` — ungated it flipped a sub cross's `end_reason`
`next_cycle→new_cycle` (subs obsolete their crosses via `_m15_create_cross`).

#### 11b Part A — Bounded multi-cycle cross capability on main (DONE 2026-06-17, byte-identical)
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

#### 11b Part B — in-flight POI resolver generalization (§10.7) — DEFERRED 2026-06-17
The MS in-flight resolver (`compute_poi_inners_for_cycle`, cycle-1-only) must generalize
to the full walk so MS's proximity-based CTS confirmation agrees with the downstream
cross for cycles ≥2. **Deferred because no main cycle≥2 cross forms on the current
window** → there is nothing for `/compare` to exercise/validate, and the change is a
substantial MS-protocol extension (MS would need per-cycle BOS/CTS maps + `P_rev`).
**Bounded gap:** it only bites when a main cycle≥2 cross actually forms AND that cycle's
CTS confirms via POI proximity — neither happens here. Revisit when a window (e.g. via
incremental config-window expansion) produces a main multi-cycle cross; then implement +
`/compare`-validate Parts A and B together.

#### 12.1 Worked validation case — the `M = 2` anchor (11b ground truth)

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
- **sid=1 CTS anchors** (clear ⇔ `CTS ≤ 0.5736` for sd=−1):

  | cycle | CTS anchor price | vs 0.5736 | clears? |
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

### 14. Decision log

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
11. **PLAN_I S3 (2026-10-08):** the sub path's crosses are bounded by the structure's cross-fib LIMIT (§2.2; the
    user's rule, `plans/PLAN_I_cross_cycle_fib_revamp.md` §1.4); a cycle past it is a labelled single. PLAN_I S4
    moves main onto the sub path and rewrites this spec (the Scenario 1 / 2 / 3 logic, the revert and the ceiling
    on `P_rev` into a "Retired logic (history)" appendix).

---

## Appendix A, part 2 — MarketStructure's retired in-flight read (history)

> Moved VERBATIM by PLAN_I S5 (2026-10-08) from the docs as they stood at `0de98ca` (PLAN_I S4) — the last commit whose
> MarketStructure ran this read (`MarketStructure._refresh_poi_inners_for_cycle` → `poi_zones.compute_poi_inners_for_cycle`
> → `fib_tracker.select_fib_anchor_for_cycle`, with MS's cycle-0 snapshot `cycle0_data` / `_update_cycle0_data` and
> the routine's "snapshot" fill-as-of policy; `git show 0de98ca:<path>`). Cross-references point into those old
> documents; headings demoted. Nothing here describes the current code (§8).

### From `zones/CROSS_CYCLE_FIB_SPEC.md` (§8 at `0de98ca`)

#### 8. MarketStructure's in-flight read (until PLAN_I S5)

MS's sd-zone-proximity CTS confirmation reads POI inners through `poi_zones.compute_poi_inners_for_cycle`, which still
selects the anchor with the retired Scenario-2 rule (`select_fib_anchor_for_cycle`: "intra" for sid 0 — every sub —
and every cycle ≠ 1) and `evaluated_at=None` (the M1 divergence, LANDMINES "Scenario 2 anchor agreement"). The drawn
fib and MS's read therefore differ where a cross or the limit decides (PLAN_I §9.2 F9: 20 / 371 refreshes on the window,
0 / 475 proximity answers changed). S5 replaces it with an embedded tracker running THIS logic (PLAN_I §9.3 D5); the
appendix then gains MS's retired read.

### From `structure/MARKET_STRUCTURE_SPEC.md` (the POI-inner refresh, "Snapshot fib on SUBS", "Snapshot vs per-candle — imbalance knowability" at `0de98ca`)

The proximity check picks the closest-to-current-price sd inner across
BOS and POIs. POI inners are refreshed at `CTS_ESTABLISHED` (new cycle)
and at each `CTS_UPDATED` (CTS extended → Fib bounds expand → IC
candidates may shift). Since PLAN_H landing B (2026-10-06) the snapshot
holds 0 or 1 POI inner: the institutional-pattern pick of the AS-OF fib
(`poi_zones.compute_poi_inners_for_cycle` → `select_ip_ic`, the drawn
POI's own selector — D2; before it, the inners of the 1–3 V30 / V60 / V90
picks). No pattern → the BOS inner alone. An IP error is NOT swallowed
(LANDMINES "The POI Resolver Fails Loudly"): only the anchor selection +
the fib construction keep the resolver's graceful `[]`. The snapshot is per-cycle in
`MarketStructureState.poi_inners_for_cycle`. Both refreshes still run
on cycle 0 (cheap; the cycle-0 POI refresh also keeps `cycle0_data` —
the Scenario-2 cond2 mirror — in sync via `_update_cycle0_data`), but
the per-candle check is gated off. The snapshot's ONLY reader is that
check (`_maybe_confirm_cts_via_proximity` → `_check_proximity_at_candle`).

**Snapshot fib on SUBS ≠ the drawn sub fib (found 2026-10-01; fix scheduled in `IDEA_PARKING_LOT.md` §D.1 item 2,
user 2026-10-02 — PLAN_I S5; since PLAN_I S4 the same holds for main, whose drawn fib also runs the one logic):** the resolver's fib choice (`select_fib_anchor_for_cycle`) returns `"intra"` for `sid == 0`, and
every sub is sid 0 — so a sub's snapshot always reads the INTRA-cycle fib (BOS_n → CTS_n), while its drawn POIs come
from the cross-cycle fib. They differ on 8 unique sub cycles of the reference window (e.g. confluence sub 0 cycle 2:
drawn POI IC 678 at 0.56234–0.56268 vs the snapshot fib's zone 0.57244–0.57364); no outcome diverged there. Detail:
`plans/PLAN_H_institutional_pattern_poi.md` §3 ("Drawn POI vs MS snapshot", D5).

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
until the next refresh), and it would have broken cond2 agreement with FibTracker's
equally uncut cycle-0 cache (retired in PLAN_I S4). FibTracker, which decides once per event, does cut
at the event's moment — the accepted MS/FibTracker divergence that follows (M1)
is documented in LANDMINES "Scenario 2 anchor agreement".

### From `LANDMINES.md` ("Scenario 2 anchor agreement" at `0de98ca`)

**Scenario 2 anchor agreement (closed 2026-05-13) — FibTracker side RETIRED by PLAN_I S4 (2026-10-08); the whole
entry retires with MS's in-flight resolver in S5.** Since S4 FibTracker no longer calls `select_fib_anchor_for_cycle`
(main runs the one logic, `zones/CROSS_CYCLE_FIB_SPEC.md`; its Scenario 1 / revert / cycle-0 cache are gone): only
MS's in-flight resolver still reads that selector, so MS's read and the drawn fib differ wherever a cross or the limit
decides (PLAN_I §9.2 F9 — 20 / 371 refreshes on the reference window, 0 / 475 proximity answers changed). The
paragraphs below describe the pre-S4 pairing (FibTracker's sentences are history):
MarketStructure's in-flight POI snapshot
(`zones/poi_zones.py::compute_poi_inners_for_cycle`) and FibTracker's
downstream POI derivation both pick Fib anchors via the shared pure
utility `zones/fib_tracker.py::select_fib_anchor_for_cycle`. The Scenario 2
cross-cycle cond1/cond2/cond3 check is, since §11a (2026-06-17), DELEGATED to
the shared pure routine `zones/cross_cycle_fib.py::resolve_cross_cycle_eligibility`
(single-step `target=1`, `fill_as_of="snapshot"`). Fill horizons are in lock-step
too: cond1's is the triggering event's MOMENT in both layers (FibTracker: the
handled event's; MS: `_refresh_poi_inners_for_cycle(moment_idx)` — Plan E E3a),
cond2 (the cycle-0 caches: the moment of their last write) and cond3
(`snapshot_horizon_idx` = the BOS_1 moment == the CTS_1 ESTABLISHED moment) since
Plan E E3a′ — any change moves both layers in the same change.
`select_fib_anchor_for_cycle`
is now a thin wrapper applying only the main-only Scenario-1 outer gate around
it. So for `sid >= 1, cycle_id == 1` cases both layers agree on whether to
anchor at `(BOS_0, CTS_1)` (Scenario 2) or `(BOS_n, CTS_n)` (Scenario 3 / intra)
— except the knowability divergence (M1) below.
MS tracks the cycle-0 snapshot (`MarketStructureState.cycle0_data`)
populated at cycle-0 CTS_ESTABLISHED, refreshed on each CTS_UPDATED, and
locked at CTS_0 CONFIRMED — threaded into the resolver via the
`PoiInnersResolver` protocol's `c0_data` slot. Its `has_unfilled` and
FibTracker's cycle-0 cache (`_cross_cycle_data[sid]["cycle0"]["has_unfilled"]`)
are both Scenario-2 **cond2** and both stored **UNCUT** (`evaluated_at=None`,
Plan F): their only reader is `select_fib_anchor_for_cycle`
(`prior_cached_liveness`) at cycle 1 only (sid ≥ 1) — FibTracker at CTS_1 ESTABLISHED, MS
at each cycle-1 refresh — i.e. after CTS_0, when every gap in `[BOS_0, CTS_0]`
has formed. Keep BOTH uncut or cond2 diverges. A FibTracker decision taken
at a write event itself — the Scenario-1 cycle-0 activation — uses the value
cut at that event's moment (at CTS_0 ESTABLISHED `_on_cts_established`'s own
`has_unfilled`; on update `FibTracker._c0_has_unfilled_now`), never the cache.

**Documented exception — the knowability divergence (M1, accepted
2026-09-24, Plan F):** FibTracker asks every decision read at the handled
event's moment (`evaluated_at` = `event_fields.event_moment(ev)`; its two
cycle-0 cache writes stay uncut); the MS
in-flight resolver passes `evaluated_at=None` (its snapshot is read only at
`i > st.cts.idx` — MARKET_STRUCTURE_SPEC "Snapshot vs per-candle"). cond2
(cached, uncut in both) and cond3 (its window ends at CTS_0, before any moment)
still agree; **cond1 and the fib's existence can diverge**. Drop case: the ONLY
sd gap of a lag-0 CTS_ESTABLISHED or a raw CTS_UPDATED has c2 == the event
candle → FibTracker creates no fib at that event, while MS — which builds its
fib as always-active in `compute_poi_inners_for_cycle` — keeps a POI inner,
usable for sd-proximity CTS confirmation from the next candle. Permanent on H1
cycle ≥ 1 (fib activation is one-shot at EST); on subs until a later
CTS_UPDATED re-asks. **0 cases on the reference window.** The same class
already existed: a gap that forms after an H1 cycle-≥1 EST enters the MS inners
at the next CTS refresh but never activates the downstream fib (not measured).
Pinned by `tests/test_imbalance_c3_knowability.py::test_m1_ms_inflight_keeps_the_inner_fibtracker_creates_no_fib`
(flips when the "re-ask when the gap's c3 closes" follow-up lands, PLAN_F §7).
Canonical: IMBALANCE_FILL_SEMANTICS.md "Knowability — the c3 rule".

**Documented approximation:** MS does NOT track Scenario 1 (TRUE / revert
to FALSE on BOS_1 touching prev BOS zone outer). It passes
`scenario1=None` in c0_data, which routes the utility into the Scenario
2/3 evaluation path. FibTracker passes the post-revert `scenario1` value
through. When Scenario 1 is TRUE in FibTracker's view (rare — requires
the CTS_0 event's moment >= reversal_confirmed_idx) AND Scenario 2 cond1/cond2/cond3 all
match (also rare), MS would pick cross-cycle anchors where FibTracker
picks intra. To plug that gap, thread the parent reversal_confirmed_idx
and prev BOS outer/sd through `structure_engine._make_market_structure`,
mirror the Scenario 1 resolution + revert logic on MarketStructureState,
and set `c0_data["scenario1"]` accordingly before each resolver call.

**Concrete signature of regression (keep for future debugging):** a
CTS_CONFIRMED dot whose `confirmation_method == "sd_zone_proximity"` BUT
the candle is visually nowhere near any rendered POI zone for that cycle
— suspect either the resolver has gone out of sync with FibTracker's
anchor selection, or `c0_data` isn't being threaded correctly through
`_refresh_poi_inners_for_cycle`. Trace by instrumenting
`_check_proximity_at_candle` to log the active `poi_inners_for_cycle` and
compare against `df.attrs["poi_zones"]`.
