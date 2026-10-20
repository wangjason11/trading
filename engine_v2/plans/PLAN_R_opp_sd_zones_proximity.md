# PLAN_R — the opp_sd zones proximity trigger: proximity reference ladders, the narrow proximity-reference cap, the proximity overlay (the user, 2026-10-14 → 19)

**Status: BUILD UNDER WAY (2026-10-20, the build session) — §8 is the build log.** Every decision closed 2026-10-20: the user's round-9 decisions ("A hold, keep 0a, go ahead with the review"); the plan review DONE and folded in (§6); Q1 = (a), Q2 = (b), Q3 = (ii), Q4 = 4 commits (§7).
- **Baseline:** **S_Q** `artifacts/commits/week9-entries-decisions/20261014_170919_da8c15e` (= `LATEST_week9-entries-decisions`, source `da8c15e`).
- **HEAD at drafting:** `bd3bf66`.
- **Suite:** 1743.
- **Branch state:** 89 ahead of origin, NOT pushed.

**Scope.** Session order item 3 (register `IDEA_PARKING_LOT.md` §D "SESSION ORDER"), the user's opposite-direction changes after PLAN_Q.

**Inputs:**
- the user's write-ups, verbatim: `qa_opp_sd_zones_inputs/user_writeup_20261014.md` (round 1) and `user_writeup_round4_20261019.md` (round 4);
- every question, answer and measurement: `qa_opp_sd_zones_inputs/qa_findings.md` (rounds 1–8);
- the measurement scripts and logs: `qa_opp_sd_zones_inputs/sims/` (README there; copied from the session scratchpad). The reference implementation of §1 is `sims/review_measure/m4_sim.py`; the shadow run is `sims/r5_shadow.py`.

**One cause per step:** four build steps, each with its own variant replay and `/compare`. The user chooses the commit granularity. The order was approved by the user (2026-10-19, "Sounds good"): 0a → 0b → 1 → 2.

---

## 1. The rules (the user's decisions, consolidated)

### Names

The user asked for "proximity" in every name: "since we have used the term reference in other places".

| Term | Meaning |
|---|---|
| **sd proximity reference** / **opp_sd proximity reference** | The zone inner each side triggers on. |
| **proximity reference ladder** | A side's candidate zones, ordered by closeness to the other side. |
| **passed proximity zone** | A ladder zone that price has closed beyond. |
| **proximity reference gap** | The SIGNED distance between the two references. |
| **narrow proximity-reference cap** | The new rule, `DEFAULT_MIN_PROXIMITY_REFERENCE_GAP_PIPS`. |
| **proximity overlay** | The sub zones added to a side. |
| **opp_sd zones proximity trigger** | Was "CTS zone proximity trigger". |
| narrow-cycle Rules 1–3 | Unchanged: the BOS–CTS threshold gap. |

### The ladders, per parent cycle, scanned on H1 candles as today

**sd side.**
- **Rungs:** the parent POI (if any), then qualifying same-direction sub zones, then the parent BOS (the LAST rung).
- **Same-direction sub zones qualify only when** the parent has a POI **and** the zone's inner lies beyond the parent POI inner, in the pullback direction.
- **No POI:** the sd side is the parent BOS alone.
- **Unreachable zones:** a zone beyond the parent BOS can never be reached, because the BOS locks first.

**opp_sd side.**
- **Rungs:** the counter-sub zones (BOS and POI), then the parent CTS (the LAST rung).
- **No other limits** on which counter-sub zones count.

**Which sub feeds a side.**
- **The sub = the side's proximity source sub, STICKY (the hold, round 9: "A hold").**
  - It starts empty (no sub rungs).
  - **Candidates:** a unique sub of that side's direction under this parent cycle, from either lens, with a NON-zero-length record under (S, C) whose start ≤ `LOH(h−1)`. The user: "based on unique subs".
  - **Moving forward:** at each candle the source moves to the newest candidate that is newer than the current source and has at least one usable zone. On the sd side that zone must also QUALIFY (beyond the parent POI); otherwise a newer sub with only non-qualifying zones would displace held qualifying rungs (plan review). Explained to the user 2026-10-20 with (1,2)'s data; see §5.3.
  - **"Newer"** = a later unique-sub `start_idx`; ties go to the higher pool idx.
  - It never moves back, and it does not change because its sub ended or was replaced, by reversal or otherwise. Option (A)'s text: "never because their sub ended".
  - **An ended source sub is FROZEN.** It contributes only zones whose moment is ≤ its lifecycle end, and outer steps up to that end; passes still apply. Its geometry keeps forming cycles to the data edge that never existed in its real projection (plan review: pool#3 c3, pool#5 c1).
  - So a sub's zones leave the ladder only by being passed, or superseded by the source sub's newer cycle or a newer source sub.
  - Why: the round-8 default ("the newest live sub with a usable zone, else the parent CTS") dropped to the parent CTS while a just-started sub had no zone yet, which restarted the ping-pong (§5.3).
- **The cycle (STRICT, round 1 Q2: "only its most recent cycle whose BOS is confirmed by then"):** the source sub's most recent cycle whose BOS zone is usable. Passing that cycle's zones never brings an older cycle back. The side then falls to the parent's last rung. On the opp_sd side a newer zone can re-open it (§1 Locking, Q2 / Q3).
- **Usability:** a zone is usable once known before the H1 candle opens. That means its M15 moment is at or before the last M15 candle of the previous hour, and the record has already started.
- **A sub POI:** usable from its cycle's CTS_CONFIRMED (final; the user: "a is correct"). It is gated on that EVENT (`ef.event_moment`), never on the POI's presence: pool#7 c2 has a POI but no CTS_CONFIRMED.
- **A sub BOS zone:** usable from its BOS_CONFIRMED event moment, never from the zone's `confirmed_idx` (clamped, and different in a floor-less view).
- **Qualifying (sd side):** a sub zone's inner STRICTLY beyond the parent POI inner. The parent POI is the cycle's sd POI rung, the one nearest the opp_sd side.

**Collapsed parent cycles, and runs without the M15 pass:** parent zones only. The user: "parent zones only for now and consider subs later". The register §G line has been added.

**The reference.** The closest usable, unpassed rung on each side ("the smallest gap available").

**Tie order** (equal inners): the parent zone first, then sub zones; within one sub, BOS before POI.

### Passed zones (round 5: "Yes")

- **The test:** an H1 close strictly beyond a zone's outer edge. Such a close has necessarily crossed the inner.
- **Which outer:** the outer as of the candle's CLOSE, including expansions during the candle (M15 steps with moment ≤ `LOH(h)`). Outers expand: H1 CTS zones and M15 BOS zones. Inners never move: 0 of 39 zones in S_Q. This is causal because the pass takes effect only from `h+1`. The sweep runs candle `h`'s pass test at the start of tick `h+1` (§5.2).
- **Effective:** from the next candle; the trigger candle's own close counts.
- **What is tested:** every candidate except the parent's LAST rung (the BOS or the CTS), which is never tested. Literally testing it would remove the last rung (measured: closes beyond the (0,0) / (1,0) / (1,1) CTS and the (0,1) BOS inside their windows) and leave a side empty. One close can pass several zones.
- **On every candle:** the pass test runs even on candles where Rules 2/3, the cap or alternation block the trigger.
- **How long:** a passed zone stays out for the rest of the parent cycle.
- **From when — DECIDED 2026-10-20, (a)** (the user: "ok let's go with a)"). A zone's closes are tested from the candle it becomes usable; for a parent zone, that is the scan start.
  - **Why the change:** the round-4 default ("closes count from the scan start, even for a sub zone that only becomes usable later"; the user then said "Yes") made almost every sub zone passed before it existed, because a sub's zones sit where price came from.
  - **The example:** sub_3 c2 POI (outer 0.58118, usable from H1 772) had closes above its outer at 683–706 and 726–729. Under the old default the overlay fires nothing on this window.

### Locking (round 7: lock by passing only)

- A side locks when its current reference is passed and the next pick is the parent's last rung (the BOS or the CTS). The next pick is made at `h+1`, after that tick's overlay refresh.
- Being the reference by default does NOT lock. Lock-by-default would kill the overlay: (0,1) would fire only 748, and (1,2)'s 954 would stay on the CTS.
- **The opp_sd lock — DECIDED 2026-10-20, (b)** (the user: "yes b) is correct").
  - **The rule (Q3 = (ii), the user: "yes let's go with ii)"):** a lock on the parent CTS is lifted by any counter-sub zone NEWER than the passed ones becoming usable. That is either:
    - a newer counter sub becoming the opp_sd source (once it has a usable zone); or
    - the current source sub forming a newer cycle with a usable zone.

    That zone is the new reference; the move is inward, so there is no counter reset. In practice the second case is rare: passing a counter sub's newest BOS zone usually means it has reversed.
  - **The sd lock stays permanent** (round 5 #7: passing the parent BOS ≈ reversal).
  - **Why it came up:** a counter sub usually ends by breaking its BOS-zone outer. An H1 close doing that passes the source's newest-cycle zones and locks the opp_sd side on the parent CTS. A permanent lock would block every LATER counter sub, against round 1 Q5 ("there will be another new counter that has started anyways. So I think it would be rare where the opp_sd zones will only be CTS zone").
  - **Effect on this window:** none. There is no opp_sd-side pass in (0,1) or (1,2), so §5.4 holds.
  - **Tests:** a lock on the CTS, then a newer source sub's zone re-opens; a lock, then the source's newer cycle re-opens; the sd lock never re-opens. **Mutants:** the unlock off; unlock on the opp side only by a newer sub (drop the newer-cycle case); unlock on the sd side.

### The narrow proximity-reference cap (rounds 4–7)

**The gap.**
- `proximity reference gap = sd × (opp_sd reference − sd reference) / pip`, where `sd` is the parent's direction.
- It is signed: crossed references are ≤ 0, which counts as narrow (round 7, "a) is correct").
- It is rounded half-up: `floor(g + 0.5 + 1e-9)`.

**Narrow:** gap ≤ 40 (H1) / 25 (M15) / 15 (M5). The constant is `DEFAULT_MIN_PROXIMITY_REFERENCE_GAP_PIPS`, asserted > the proximity pips.

**The counter state machine, per cycle, evaluated at the start of each candle.** The user, round 5: "Not narrow → reduce to narrow and counter resets → expansion resets counter on update side if still too narrow & reduction does nothing → expansion to not narrow & counter is gone → …"

| Transition | Effect |
|---|---|
| wide → narrow, or narrow at the scan start | both side counters start at 0 |
| narrow → narrow, a side's reference moved OUTWARD (away from the other side) | that side's counter resets to 0 (cause-agnostic: a pass, a sub change, a new sub cycle). Outward is judged on THAT side's own inner, whatever the other side did on the same candle (the user, round 5: "on the side where the update happened"). A net-gap reading would reset both at (1,2) 1048 and bring back 1048 / 1049. |
| narrow → narrow, inward move | nothing |
| narrow → wide | counters dropped |
| while narrow | a side may fire only if its counter < 1; firing adds 1 |

**Combination with today's rules.** The narrow-cycle Rules 1–3 (the BOS–CTS gap, < 50 / 30 / 15) stay unchanged. A trigger fires only if BOTH rule sets allow it. Rule 2 is NOT part of the new cap.

### Unchanged

- Alternation sd → opp_sd → sd, and one trigger per candle.
- The scan window.
- Proximity pips 8 / 5 / 3 (the code value; the user's 9 / 6 / 3 came from outdated docs, fixed in step 0b).
- The detectors (subsequent_confluence on every opp_sd past list index 0; subsequent_counter on an sd at index ≥ 2).
- The WVMI gate. The first sd per cycle is always on the first sd rung, which is always a parent zone.
  - **Why (plan review, from the code):** the first expected direction is sd. A POI is passed only by a close beyond its outer, so the low has crossed the inner and the sd fires on the POI on that candle or earlier. Rule 3 and the new cap start at 0.
  - **The dependency:** Rule 2 could block it only on a narrow cycle whose CTS was not confirmed by a pullback, which Rule 1 rules out (`market_structure.py:1907-1917`). The assert documents that dependency.
- MarketStructure's own sd-proximity CTS confirmation stays on parent zones. The user, round 8: "this would also ensure that CTS confirmation does not consider these subs".

---

## 2. Step 0a — the sibling-CTS fallback window starts at the parent cycle (zero change)

**The bug (latent).**
- `entity_df_mutation._sibling_cts_idx_window` gives `first_counter` the bound `lo = 0`. A `subsequent_*` trigger whose prior-proximity meta is missing degrades to `lo = 0` too.
- The code comment calls this safe because "the pool read is already scoped to this parent cycle". That holds for the pool query.
- It does not hold for the FALLBACK, `_window_extreme_idx(m15_df, lo, hi, −lower_sd)`, which scans the raw M15 frame. With `lo = 0` the fallback picks the lowest low (or highest high) since the data start.
- Measured (round 2; re-measured at the plan review): (1,2)'s first_counter with no sibling would pick M15 385 (in H1 96; 2025-11-20 22:15 UTC, low 0.55808) instead of M15 3621 (0.57112), 130.4 pips lower. (The draft said "about 146 pips", which was wrong.)

**Reachability.**
- **S_Q:** not hit (0 `sibling-CTS unavailable` lines in the S_Q `run.log`).
- **This change (the shadow, §5.4): the "sub_4" path does not hit it.**
  - The earlier claim that it "becomes live" was WRONG, and is corrected here.
  - In the new run, (1,2)'s first_confluence re-finds "sub_4" through an APPROX probe-cache hit (finalize 3491). Its record therefore starts at max(3491, 3611, floor 3611) = 3611 = the sub's (0,1) parent_end, which is the same-candle handover. The sub stays live into (1,2), so the sibling read finds it.
- **But step 2 DOES reach the fallback once, by another path** (plan review, `shadow_full2.log:8639`).
  - The trigger: 1046's subsequent_counter (1,2), trigger M15 4187, window (4183, 4187).
  - Why it falls back: its sibling, the sub started by 1045, has 0 events, so there is no CTS. The ad-hoc fallback picks 4183, which becomes the new sub's `starting_idx`.
  - Step 0a does not change it: lo 4183 > floor 3611.
  - The draft's "not hit either" was WRONG. Step 2's `run.log` is predicted to have 1 `sibling-CTS unavailable` line.
- **Still reachable in other windows:** a first_confluence that re-finds an ALREADY-ENDED sub with a fresh probe finalizing after the handover candle gets a zero-length record. That cycle's first_counter then has no sibling.
- **Decision:** kept as a cheap zero-change fix (round 9: "keep 0a").

**The bound — REVISED 2026-10-20 at the landing review (the user: "Establishment moment"; `qa_findings.md` round 13):**
`lo` is bounded by the parent cycle's CTS-established MOMENT on M15 (`_loh(tables.cts_moment[(S, C)])`, the parameter
`m15_parent_cts_established_idx`), NOT the real-time floor this section first named. The code reviewer found that the
floor empties the window (`lo > hi`) for a trigger fired before it — a retroactively established cycle that still has
a real-time lifecycle (example: had (1,2) been established at H1 910, (1,1)'s 820 / 871 would get `[3611, 3283]` /
`[3611, 3487]`; with the moment `[2995, 3283]` / `[3283, 3487]`). Every trigger comes after its cycle's CTS_CONFIRMED
≥ that moment, so the window is never empty (asserted). On S_Q the two coincide on every live cycle ((0,0) 115, (0,1)
652, (1,2) 902) → still zero change. The text below keeps the original "floor" wording as the plan record.

**The code.**
- `_resolve_trigger_m15_start` → `_resolve_sibling_cts_via_unified_probe` → `_sibling_cts_idx_window` gain a REQUIRED `parent_floor_idx` (M15). The orchestrator's `_resolve_start` closure passes `tables.floor(S, C)`, the same floor every TriggerRecord already stores.
- Then `lo = max(lo, parent_floor_idx)` for every use case. The comment is rewritten to say the pool query and the raw-frame fallback have different scopes.

**Prediction:** S_Q byte-identical. 0 / 27 CSVs, 0 hover leaves. The `run.log` grep for `sibling-CTS unavailable` stays at 0.

**Tests:**
- `_sibling_cts_idx_window` with a floor: first_counter `lo == floor`; subsequent_* `lo == max(prior hour end, floor)`; a missing prior-meta → `lo == floor`.
- A fallback case on a synthetic frame whose global extreme lies BEFORE the floor: the fallback picks the in-cycle extreme.
- The real collaborator: the shadow pin `(1,2)` window `lo 0 → 3611` (the fix0a shadow log).

**Mutants:** drop the `max`; apply it to first_counter only; pass `trigger_idx` instead of the floor; and the call site (the closure passes `0`).

**Docs:** PART4 §4.3.3–§4.3.5 (the step-4 fallback's window) and the `_sibling_cts_idx_window` docstring.

---

## 3. Step 0b — MarketStructure's fallback proximity pips 9 → 8 (zero replay change)

**The finding.**
- `market_structure.py:91` `_FALLBACK_PROXIMITY_PIPS = 9`, labelled "matches H1 default". `DEFAULT_PROXIMITY_PIPS["H1"]` has been 8 since `804d19d` (2026-05-28).
- It is reached only by callers that build `MarketStructure` directly: 7 test files, 20 constructions, measured by recording `proximity_pips=None` across the full suite. The draft said 9; two of those files only name it in docstrings.
- Every production build passes the value: the 3 call sites, `structure_engine.py:198,395` and `unified_probe.py:541`, all go through `setdefault`.

**The code.**
- `_FALLBACK_PROXIMITY_PIPS = 8`.
- A guard test: `_FALLBACK_PROXIMITY_PIPS == DEFAULT_PROXIMITY_PIPS["H1"]` and `_FALLBACK_MIN_GAP_PIPS == DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS["H1"]`. This replaces the "kept in sync manually" comment with a check that fails loudly.

**Prediction.**
- S_Q byte-identical.
- The 9-file set: 242 / 242 pass at 8 (measured by patching the module global).
- The full suite at 8: 1743 / 1743 (plan review).
- The full suite is run at the build; any change in outcome is reported, not absorbed.

**Mutant:** the constant back to 9, which the guard must catch.

**Docs, already applied in the working tree (round 5):**
- the LANDMINES "Concrete cases" note;
- 11 comments in `tests/test_zone_proximity.py`;
- 2 memory notes;
- the register lines (§E the probe's H1-pips-on-M15, inert by design; §G would-have-been subs for collapsed cycles);
- `qa_findings.md`.

They ride with this step's commit (agreed 2026-10-19).

---

## 4. Step 1 — the proximity reference ladders and the narrow proximity-reference cap, on parent zones (zero change)

### The code (`zones/zone_proximity.py`)

- **Constant:** `DEFAULT_MIN_PROXIMITY_REFERENCE_GAP_PIPS = {"H1": 40, "M15": 25, "M5": 15}`, with an import-time assert > `DEFAULT_PROXIMITY_PIPS`.
- **Refactor the per-cycle loop into one per-cycle scan object.** For example `_CycleProximityScan`, with `step(i) -> Optional[ZoneProximityTrigger]` and `close(i)`.
  - **State:** the two ladders; the passed set; the locks; the previous references and gap; the counters; `expected_dir`; plus today's Rule 2/3 state.
  - **Why an object:** step 2 drives the SAME object from the sweep. Single source of truth: the batch scan and the sweep scan never fork.
- **Step 1's ladders:**
  - sd = the cycle's POI(s) closer than its BOS inner, then the BOS (last rung). A POI beyond its BOS is unreachable, which is today's "closer of the two" behaviour (round 6 #7).
  - opp_sd = the CTS (last rung).
- **Per candle `i`:**
  1. Advance the threshold timeline (today's code).
  2. Pick the references.
  3. Compute the signed gap and narrow state, and run the counter state machine (§1).
  4. Apply Rules 2 and 3 (unchanged).
  5. Apply the new cap.
  6. Run the trigger test against the reference (today's comparisons).
  7. Then the pass test on `close(i)`, applied from `i + 1`.

  Steps 4–5 (and alternation) gate ONLY step 6. Steps 1–3 and 7 run on every candle. Today's `continue`s (`zone_proximity.py:484-490`) must not skip the pass test.
- **`ZoneProximityTrigger`:**
  - `zone_kind` and `trigger_inner` come from the reference, which equals today's values on parent zones.
  - The meta gains `prox_ref_gap_pips`, `narrow_prox_ref` and `prox_ref_source` (`"parent"`).
  - The hover does not read meta, so no figure change.
  - The docstring drops "the only opposite-direction zone, by definition".
- **The rename:**
  - the internal trigger meta keys and locals in `multitf/` (`prior_cts_prox_idx` → `prior_opp_sd_prox_idx`, `this_cts_prox_idx`, `cts_prox_idx`, `cts_prox_zone_kind`, `cts_prox_t`);
  - the docstrings;
  - PART4 §4.3.4 / §4.3.5 wording.

  None of these is saved: they are not in a CSV or a hover (verified by grep: `multitf/` only). The parent-CTS hover label `opp_sd:CTS` stays accurate.

### Prediction (simulated, `r4_parent_only.py`): S_Q byte-identical

| Cycle | Triggers (unchanged) | What the new rules do, invisibly |
|---|---|---|
| (0,0) | none | sd = the BOS only (no POI) |
| (0,1) | 748 | the POI is passed at 754 (close 0.57657 < outer 0.57676), so sd = BOS 0.57494 from 755, locked; the gap stays wide |
| (1,0) | none | gap 30 from the start: NARROW (nothing reaches) |
| (1,1) | 820, 871 | gap 59 |
| (1,2) | 935, 954, 1035 | the POI is passed at 1047 (close 0.57852 > 0.5785), so BOS from 1048 |

Expected: 0 / 27 CSVs and 0 hover leaves. Chart counts H1 81/236, M15 146/135 and 289/293 unchanged.

**Why the passes stay invisible: alternation, not price** (plan review). After each cycle's last sd trigger, the opp_sd never fires, so the sd reference is never consulted again:
- (0,1): the highest high in 749–901 is 0.58105, short of the 0.58324 the CTS needs.
- (1,2): the lowest low in 1036–1057 is 0.57598, short of 0.57362.

In (0,1), 106 lows in 755–901 come within 8 pips of the passed POI. Test 10 and the docs say this.

An independent implementation by the rules reviewer, on the captured `check_zone_proximity` inputs (threshold timeline, Rules 2/3, cap, locks), gives the same 6 triggers under all 5 readings tried. The cap binds only in (1,0), where nothing reaches.

### Tests (synthetic frames, `tests/test_zone_proximity.py`)

1. A pass moves sd POI → BOS: the next sd fires on the BOS level, not the POI.
2. A close exactly AT the outer is not a pass.
3. Lock: after POI → BOS, nothing moves the sd reference again.
4. A POI beyond its BOS is never used.
5. The cap:
   - the counters start at the narrowing;
   - triggers before the narrowing don't count;
   - an outward move while narrow resets only that side;
   - an inward move doesn't reset;
   - narrow → wide → narrow restarts the counters.
6. A signed crossed gap (≤ 0) counts as narrow.
7. Rounding: 40.5 → 41 is wide; 40.4 → 40 is narrow.
8. Both rule sets AND-ed: structural narrow + reference wide → Rule 3 still caps.
9. The tie order.
10. The real collaborator: S_Q's (0,1) / (1,2) parent ladders, using the saved zones from the S_Q save, reproduce today's 6 triggers and the two pass candles 754 / 1047. Alternation is what keeps the passes invisible.
11. Both references move on one candle (one outward, one inward): only the outward side resets.
12. The last rung is never passed: a close beyond the BOS / CTS outer leaves it the reference.
13. A pass on a blocked candle (Rule 3, the cap or alternation) still counts.
14. A cycle narrow from the scan start starts its counters at 0.

### Mutants

Every rule above inverted or dropped:
- `<=` → `<`;
- the reset on inward moves;
- no reset at narrowing;
- `abs` instead of signed;
- the pass on the close vs on the low;
- the pass applied on the same candle;
- lock-by-default;
- the reset judged on the net gap instead of per side;
- the last rung tested for a pass;
- the pass test skipped on a blocked candle.

Every mutant is run, including any called "equivalent".

### Docs (same commit)

- `zone_proximity.py` module docstring;
- PART4 §4.3.2–§4.3.6 (the proximity scan);
- LANDMINES "Zone proximity gate" + a new "Proximity Reference Ladders" entry;
- GOTCHAS (start-of-candle evaluation for the cap);
- GLOSSARY (the new terms + the three clocks: anchor / moment = "would have been" / lifecycle = real time — round 4 #9);
- the register §D item 3 line.

---

## 5. Step 2 — the proximity overlay, inside the lifecycle sweep (the behaviour change)

### 5.1 Why inside the sweep (round 1, 7a)

The overlay is circular: opp_sd and sd triggers read sub zones, and subs come from triggers. Today the scan runs ONCE, in `run_pipeline` (`orchestrator.py:422`), before `_run_multi_tf_dual`. The sweep (`multitf/lifecycle_sweep.py`) takes the whole trigger list up front.

**What stays precomputed:**
- first_confluence (from BOS events);
- first_counter (from the WVMI gate, the first sd per cycle);
- reversals.

The first sd per cycle is always on a parent rung, so first_counter cannot change. This is ASSERTED against the incremental scan.

**Only subsequent_confluence / subsequent_counter become sweep-generated.**

### 5.2 Design

**A new moment kind `proximity_tick` per non-collapsed parent cycle and H1 candle `h` of its scan window.**
- **When:** at M15 `LOH(h−1)`, in a NEW phase that sorts AFTER `SUB_END` (revised at the plan review; the draft had `LOH(h−1)+1` before FIRE). All pool moments ≤ `LOH(h−1)` are therefore processed: "known before the H1 candle opens".
  - **Why:** the knowability is identical, and the tick no longer needs `LOH(h−1)+1` to exist. Measured: H1 1057 has NO M15 bar (the M15 frame ends at 4227; the M15 fetch ends at the last H1 row's open time, `orchestrator.py:920-923`). (1,2)'s scan runs through 1057.
  - **Ordering:** ticks of different cycles at one idx sort by (S, C), then `h`.
- **What it does:**
  1. Run candle `h−1`'s pass test, on `close(h−1)` with outers as of `LOH(h−1)` (§1 "Which outer").
  2. Refresh each side's source sub and overlay rungs from the pool (§1; frozen at an ended source's end).
  3. Run the SAME `_CycleProximityScan.step(h)` as step 1 on candle `h`'s OHLC. That is price data, not engine state; the trigger is stamped at `LOH(h)`, as today.
  4. On a trigger, run the detectors on the cycle's list. They are causal: they read only list[i−1], list[i−2] and the H1 OHLC in [prior.idx, h] (`subsequent_confluence_trigger.py:86-118`, `subsequent_counter_trigger.py:85-125`). A new subsequent_* trigger is pushed as a `SweepTrigger` at `LOH(h)`, phase FIRE, which lies after the tick, so the "never queued in the past" assert holds.
  5. **An H1 hour with no M15 bar** (`LOH(h) == LOH(h−1)`): the scan step still runs, so the H1 list matches the batch, but no sub trigger is pushed. It is logged `UNRESOLVED (skipping) reason=no_m15_hour`. Today such a batch trigger would be stamped one hour early (a pre-existing quirk, not hit in S_Q).
- **What the tick may read:** `pool.records_for(L, S, C)` for both lenses; non-zero-length records with start ≤ `LOH(h−1)`; and processed `end_idx` values. Never `_active`, which drops the ended records the hold needs. Never `natural_reversal_idx`, which is future-known from creation (`entity_df_mutation.py:1235`).
- **Collapsed cycles and runs without M15:** the step-1 batch scan (parent zones only). "Collapsed" is static and known before the sweep (`build_parent_tables`, `collapsed = floor_m15 ≥ end_m15`).

**The sub zone view.** Sub KL / POI zones today exist only after the sweep (`render_sub_projection` → `project_to_window`).
- **The view (revised at the plan review): each zone is read AS OF ITS OWN USABLE MOMENT.**
  - Per pooled sub, the geometry's events (to the data edge) give each cycle's BOS_CONFIRMED and CTS_CONFIRMED moments.
  - For each such moment, the SAME `project_to_window` runs capped at it. That yields the cycle's BOS zone (inner, outer steps with their moments) or its POI (inner, outer).
  - **Cost:** 0.02–0.09 s each, about 2 per cycle, so roughly 1–3 s on this window.
- **Why not the natural-end view the draft had:**
  1. It holds cycles formed after a sub ended (pool#3 c3 POI, CTS_CONFIRMED 3741 after its 3611 end; pool#5 c1, BOS_CONFIRMED 4200 after its 4143 replacement). The hold would read them. The fix: the §1 freeze at the source's end, plus moment ≤ `LOH(h−1)`.
  2. "A POI is final at its cycle's CTS_CONFIRMED" holds on this window but is not guaranteed by the mechanism.
     - Measured: 0 pick changes over 2,537 capped projections; 41 / 41 zones identical on a frame truncated at the usable moment.
     - Not guaranteed because the drawn pick's IC reference is evaluated at the cap (`poi_zones.py:863` → `ic_reference(evaluated_at=…)`), and `fib_tracker.py:438-508` reads `known_at=evaluated_at` and a moving P.
     - Reading as of the moment is causal by construction.
- **ASSERTED after the sweep:** every sub zone a trigger used has the same inner (and kind / cycle) in the sub's real projection. On this window it must hold. If it ever trips in another window, a zone changed after it became usable: a design question for the user, not a silent pick.

**The H1 trigger list.**
- **What replaces what:** the sweep's final per-cycle lists replace `attrs["zone_proximity_triggers"]` and the meta's subsequent_* lists.
- **Main WVMI:** unchanged, and asserted. The gate and the first-sd stream read only `trigs[0]` (`_first_sd_prox_gate`, `orchestrator.py:113-127`). The assert compares (idx, direction, trigger_inner, zone_kind) of each cycle's first sd between the batch and the sweep; the new gap meta differs, so it is not compared.
- **Every other pre-sweep reader switches to the final lists** (plan review, `orchestrator.py`):
  - the sub WVMI attribution `_wvmi_trigger_streams_by_lens(..., var3_all, var4_all)` (1071-1073), which runs after the sweep on the step-1 detector lists today;
  - `s_res.df.attrs["subsequent_*_triggers"]` (607-609, 627-629);
  - the `[multi_tf:dual] … var3= var4=` count log (908-913);
  - the early return when no triggers exist (914-916), which must no longer depend on var3 / var4.
- **The sweep input** (969-988) keeps:
  - first_confluence, first_counter and reversals;
  - plus ONLY the collapsed cycles' batch var3 / var4, which produce the `collapsed_parent_cycle` unresolved rows (S_Q: `confluence,1,1,subsequent_confluence,3487`).

  The merged lists are re-sorted by `trigger_event_idx`.

**Trigger meta and the hover.**
- **The meta keys** (separate keys, no mixed-type field):
  - `prox_ref_source` ∈ {`"parent"`, `"sub"`};
  - `prox_ref_sub_id`, mapped from the pool idx after `number_subs_by_start`, `lifecycle_sweep.py:535-542`. A source sub always has a started record, so it always gets an id;
  - `prox_ref_cycle`;
  - `zone_kind`.
- **The hover type line** names the source, for example "opp_sd: sub_3 c2 POI 0.58096" (round 7: "Sounds good"). It changes in BOTH charts: the H1 chart (`export_plotly.py`) and the M15 lens overlay (`export_m15_chart.py:2669-2831`), which has the same lookups.
- **`zone_confirmed_idx`:** the KL / POI lookup by (parent sid, cycle, kind) would mislabel a sub zone.
  - The POI branch is the dangerous one: `_poi_activation_floor_idx` picks the nearest parent POI of (sid, cycle), so an `opp_sd:POI` from a sub would silently get the parent sd POI's floor.
  - **Fix:** branch on the source BEFORE the zone kind, in both charts. A sub-sourced trigger shows its zone's usable-from moment (M15 → the H1 candle it counts from).
  - The parked register §G item about `zone_confirmed_idx` stays separate.
- The optional stepped "current proximity reference" line per side is rendered as a SAMPLE for the chart look, not locked.

**Docs:**
- POI_ZONES_SPEC "POIs are always sd-direction (invariant)" → "Consequence" is rewritten. The invariant still holds per structure; the parent's opp_sd side now reads a counter sub's POI.
- Register §E "Opposite-direction POIs" is CLOSED.
- PART4 §17 (the sweep's new moment kind + phase table), §4.3.4 / §4.3.5;
- CHARTING_SPEC (the hover);
- LANDMINES (the overlay + its asserts).

### 5.3 DECIDED (A) the hold — a newly started sub with no usable zone yet (found by the shadow, §5.4)

The user, round 9: "A hold". The rule as built is the sticky proximity source sub in §1. The analysis that led to it follows.

The round-8 default ("the newest live counter sub with a usable zone, else the parent CTS") restarts the ping-pong in (1,2):

1. **H1 1046.** The sd trigger → subsequent_counter → a new up-direction sub (pool#10, start 4183). It replaces old sub_7 at M15 4187 (`same_dir_replacement`), but its first BOS confirms only at M15 4189.
   - The earlier statement "8 / 8 subs had BOS_0 before their record start" holds for S_Q only.
   - Two new subs break it: this one, and 1045's (−1, 4179), which has no events at all.
2. **H1 1047.** No live counter sub has a usable zone, so opp_sd = the parent CTS 0.57282. The gap is 50 → WIDE, so the counters are dropped.
3. **H1 1048.** The new sub's BOS (inner 0.57818) is usable, and the sd side moved to the BOS 0.57996 (the POI was passed at 1047). The gap is 18 → NARROW again with FRESH counters.
   - 1048 opp_sd fires (low 0.57792 ≤ 0.57898).
   - 1049 sd fires (high 0.57940 ≥ 0.57916).
4. Each fires a subsequent_* that starts yet another sub, and the next round can repeat it.

**Options:**

- **(A) Hold — recommended.** While the newest sub on a side has no usable zone yet, the side keeps the previous sub's zones, even though that sub was just replaced. The hold lasts until the newcomer's first zone becomes usable, or those zones are passed.
  - Zones then leave a ladder only by being passed or superseded, never because their sub ended.
  - (1,2) → 1045, 1046, then nothing through 1057.
    - At 1047 the reference holds old sub_7's cycle 2 BOS 0.57728 (gap 6).
    - At 1048: opp → the new sub's BOS 0.57818 (inward, no reset). The parent POI was passed at 1047, so sd moves outward and sd resets.
      - **Where sd moves, with the qualifying switch (§1):** sd → "sub_4" (now sub_5, key (−1,3304)) c0 POI 0.57885 (outer 0.5792, usable from H1 936; it qualifies: beyond the parent POI 0.57786 and inside the BOS 0.57996). Gap 7.
      - It is passed at 1050 (close 0.5794 > 0.5792), so sd = the parent BOS 0.57996 from 1051, locked, gap 18.
      - **Without the qualifying switch:** old sub_6 (now sub_7, (−1,3760)) became the sd source at H1 955. Its zones (BOS 0.57758, POI 0.57653) lie INSIDE the parent POI and don't qualify, so sub_5's POI was dropped and sd went straight to the BOS at 1048.
      - Same triggers either way (`review_measure/m4_sim.py`, src_qual True / False).
    - The opp_sd counter is still 1, so 1048 is BLOCKED (and 1049, 1050, 1054).
    - At 1049 the sd counter is 0 again, but alternation still expects an opp_sd after 1046's sd, so 1049 is blocked too.
- **(B) A narrow episode ends only through a pass** (price). A sub or zone change can move a reference but never ends the episode. Same result here: 1045, 1046.
- **(C) Keep the round-8 default.** 1045, 1046, 1048, 1049, plus the subs they start, plus possible further rounds. These can't be predicted without the build.

The same choice applies symmetrically to the sd side (same-direction subs).

### 5.4 Prediction — from shadow runs of today's code with the new triggers injected

`r5_shadow.py` takes today's pipeline and replaces the (0,1) list with the round-4 chain, and in the "full" variant also (1,2)'s. The build must REPRODUCE these numbers.

**Measured vs simulated** (plan review):
- **Simulated, not engine output:** WHICH H1 candles fire — 820, 872, the 954 relabel, 1045, 1046, and 1048+ blocked. Both chains were injected by hand.
- **Measured:** every pool outcome given those lists — keys, records, lifecycles, end reasons, probe inputs, the new subs' zones, unresolved triggers, WVMI.

**The firing chain is reproduced** by the measurement reviewer's independent §1 implementation (`review_measure/m4_sim.py`), which feeds the new subs' dumped zones back in. That holds under: passes tested from when usable (§7 Q1), the hold, the strict cycle, per-side resets, and the outer as of the close. With passes from the scan start it gives S_Q's 6 triggers and no new subs.

The predictions below are for the decided hold (A) and Q1 = "from when usable".

**H1 proximity triggers: 6 → 10.**

| Cycle | Triggers |
|---|---|
| (0,1) | 748 sd:POI (=); **+820 opp_sd: sub_3 c2 POI 0.58096**; **+872 sd: BOS 0.57494** (not 830–833: the POI was passed at 754) |
| (1,1) | 820 sd:BOS, 871 opp_sd:CTS (=, collapsed: parent zones) |
| (1,2) | 935 sd:POI (=); 954 opp_sd relabelled CTS 0.57282 → **sub_6 c0 POI 0.57306** (the same candle); 1035 sd:POI (=); **+1045 opp_sd: sub_8 c2 BOS 0.57728**; **+1046 sd: POI 0.57786** |

**Subs: 8 → 11.** Records: 11 → 15. Unresolved: 4 = 4. New ids use `number_subs_by_start`; `/compare` keys by (direction, starting_idx), not by sub_id.

| New id (old) | Key | Lifecycle | Change |
|---|---|---|---|
| sub_0 / 1 / 2 | (1,454) / (−1,1797) / (1,2365) | = | = |
| sub_3 | (−1,2639) | [2829, **3491**] same_dir_replacement by sub_5 | end was 3611 parent_end |
| **sub_4** (NEW) | (1,3230) | [3283, 3589] reversal | from 820's subsequent_confluence (probe input = sub_3's c2 CTS anchor 3047); BOS inner 0.57442, POI 0.57628–0.57658 |
| sub_5 (old sub_4) | (−1,3304) | [**3491**, 3819] same_dir_replacement by sub_7 | from 872's subsequent_counter (input 3304 = sub_4's CTS anchor). Records: counter (0,1) 3491–3611; confluence reversal (0,1) 3589–3611; confluence first_confluence (1,2) 3611–3819, live through the SAME-CANDLE HANDOVER (APPROX cache hit, finalize 3491). This is the user's round-1 expectation: "active from the starting_idx", the later first_confluence a re-trigger. |
| sub_6 (old sub_5) | (1,3621) | [3743, 4143] | = (sibling found, input 3621) |
| sub_7 (old sub_6) | (−1,3760) | [3819, **4183**] same_dir_replacement by sub_9 | end was 4200 reversal; its reversal spawn at 4200 becomes inert |
| sub_8 (old sub_7) | (1,4027) | [4143, **4187**] same_dir_replacement by sub_10 | was open-ended; its confluence reversal record (4200) is gone |
| **sub_9** (NEW) | (−1,4179) | [4183, open] | from 1045's subsequent_confluence; **0 events** (no structure before the data edge 4227) |
| **sub_10** (NEW) | (1,4183) | [4187, open] | from 1046's subsequent_counter; BOS_CONFIRMED 4189, inner 0.57818 |

**Other expectations:**
- Main WVMI records unchanged. Sub WVMI 8/18 → 9/19 (measured on the shadow).
- 1 `sibling-CTS unavailable` line in `run.log` (§2: 1046's subsequent_counter; S_Q has 0).
- H1 chart: +4 markers, 1 relabelled.
- M15 lens charts and CSVs change (sub WVMI, zones, records). Their counts are measured at the variant replay and compared against this table.

### 5.5 Tests and mutants

**Tests:**
- The sweep tick ordering: a sub zone confirmed in hour `h` is unusable at `h` and usable at `h+1`.
- Record-start gating.
- The overlay rung filters: sd only beyond the parent POI; the latest-started sub; its most recent cycle.
- The POI from CTS_CONFIRMED.
- The hold: a replaced source sub's zones stay until the newcomer has a usable zone; the source never moves back; the (1,2) 1047 → 1048 sequence on a synthetic frame.
- An ended source sub is frozen: its geometry forms a new cycle after its end, and the ladder never reads it.
- The sd source switch needs a QUALIFYING usable zone.
- Usability from the events: a POI without its cycle's CTS_CONFIRMED is never usable.
- A no-M15 hour: the scan step runs, no sub trigger is pushed, the UNRESOLVED line is logged, and there is no past-queue assert.
- The sub WVMI attribution reads the final lists.
- The M15 overlay hover branches on the source.
- The first-sd == WVMI-gate assert.
- The view == projection assert.
- The hover source line.
- The predicted table (§5.4) as an end-to-end pin on the real data path, plus a stubbed-sweep unit test in the style of `lifecycle_sweep`'s predicted-table test.

**Mutants:**
- the tick phase before SUB_END;
- `LOH(h)` instead of `LOH(h−1)`;
- drop the record-start gate;
- the POI from activation instead of CTS_CONFIRMED;
- the sd filter off;
- the hold off;
- the end-freeze off;
- the natural-end view instead of as-of-moment;
- the sub WVMI attribution on the step-1 lists;
- and the call sites.

---

## 6. Reviews and order

- **Plan review: DONE 2026-10-20.** 3 read-only agents (the rules, the measurements, the sweep integration), 0.84M tokens in total (rules 0.24M, measurements 0.27M, sweep 0.33M). Their findings are folded in place above, marked "plan review". Their scripts and logs are in `qa_opp_sd_zones_inputs/sims/review_rules/`, `review_measure/` and `review_sweep/`. The questions it raised for the user (§7 Q1–Q3) are all answered.
- **Landing reviews** (the user, 2026-10-20: "combined review is fine"):
  - **ONE combined cold review of 0a + 0b + 1,** after step 1 lands. 2 read-only agents: the code against this plan; test and mutant adequacy. About 0.4–0.6M tokens.
  - **The step-2 review,** after its `/compare`. 3 read-only agents: the code against the user's rules (§1, §7); the sweep integration (§5.2); the measurements against §5.4. About 0.8–1.2M tokens.
  - About 1.2–1.8M in total, stated beforehand. Each review's tokens are reported.
- **Order and gates:**
  1. 0a → 0b → 1, each zero-change, each with its variant replay + `/compare` (expected 0 / 27).
  2. Then 2: the variant replay + `/compare` against this prediction → mutants → the user's chart look (H1 first) → commit only on the user's go → `/commit-save`.

## 7. Decisions (the user, round 9, 2026-10-19: "A hold, keep 0a, go ahead with the review")

1. **§5.3:** (A) the hold — the sticky proximity source sub (§1).
2. **§2:** step 0a kept.
3. **§6:** the plan review: go (about 3 read-only agents, 0.5–1M tokens, stated beforehand).

**After the plan review (the user, 2026-10-20):**
- **Q1 — (a)** "ok let's go with a)": a sub zone's closes count for a pass from the candle it becomes usable (§1).
- **Q2 — (b)** "yes b) is correct": a newer counter sub's usable zone re-opens the opp_sd side after a lock on the CTS; the sd lock stays (§1 Locking).
- **Q3 — (ii)** "yes let's go with ii)": the opp_sd lock is lifted by a newer source sub's usable zone OR the source's newer cycle's usable zone (§1 Locking).
- **Q4 — "ok"**, 4 commits:
  - 0a;
  - 0b, carrying the uncommitted doc fixes, the plan, the Q&A folder and `sims/`;
  - 1;
  - 2.

  Each commit follows its own variant replay + `/compare`; 0a / 0b / 1 must show 0 / 27. `/commit-save` runs ONLY after step 2, the only step that changes outputs; a zero-change step's save would duplicate S_Q.

## 8. Build log (2026-10-20)

- **Step 0a — LANDED `a6e71c0`.** As §2: `parent_floor_idx` REQUIRED through the three functions, the orchestrator
  closure passes `tables.floor(*t.cycle)`. Prediction met: vs S_Q 0 / 27 CSVs (md5), 3 / 3 figures equal in every
  trace attribute + layout (`cmp_fig_full.py`), counts 81/236, 146/135, 289/293, 0 `sibling-CTS unavailable`, skip
  grep 22. Suite 1749 (+6: the window class's floor cases, the fallback-extreme case on a synthetic frame, the
  required-argument checks, the orchestrator call-site pin in `test_lifecycle_sweep_unit.py`; the real floors are
  already pinned in `test_parent_tables.py:346`). Mutants 7 / 7 KILLED (§2's four + the dispatcher forwarding 0 +
  the missing-meta degrade to 0). Replay 48.7 s.
- **Step 0b — LANDED (this commit; it carries the planning files: the plan, `qa_opp_sd_zones_inputs/`, the register /
  LANDMINES / `test_zone_proximity.py` comment fixes).** As §3: `_FALLBACK_PROXIMITY_PIPS = 8`, the guard
  `tests/test_ms_fallback_proximity_defaults.py` (both fallbacks == the H1 table values; a direct construction reads
  them). Prediction met: vs S_Q 0 / 27 CSVs, 3 / 3 figures equal, counts unchanged, 0 `sibling-CTS unavailable`. Suite
  1752 (+3; no other outcome changed — the plan review's 1743 / 1743 at 8). Mutants 2 / 2 KILLED (the constant back to
  9; the min-gap fallback 50 → 49). Replay 47.7 s.
- **Step 1 — LANDED (the commit after `c1bd05a`).** As §4. `zones/zone_proximity.py`: `DEFAULT_MIN_PROXIMITY_REFERENCE_
  GAP_PIPS` 40 / 25 / 15 (+ the import-time assert); `ProximityRung` (key, side, kind, inner, `outer_at(i)` = the outer
  as of candle i's close — a POI's fixed edge, a KL zone's INIT outer then each expansion from its moment
  `expanded_idx`; `last`; `source`; `tie_rank`); `CycleProximityScan` (`step(i)` / `close(i)`, `set_overlay` = step 2's
  entry point, `gate` = the gate that blocked candle i's trigger test: `rule2` / `rule3` / `prox_ref_cap`);
  `build_cycle_proximity_scans` (shared by the batch and, in step 2, the sweep); `check_zone_proximity` = the scans
  driven over their windows; `_try_trigger_at_candle` gone. The trigger meta keys `prox_ref_gap_pips` /
  `narrow_prox_ref` / `prox_ref_source`. The rename (`prior_cts_prox_idx` → `prior_opp_sd_prox_idx`,
  `cts_prox_zone_kind` → `opp_sd_prox_zone_kind`, `cts_prox_t` → `opp_sd_prox_t`, the docstrings' "CTS-prox" →
  "opp_sd-prox"; the readers' `.get` degrade-to-floor kept — a test pins it). Build notes:
  - Three existing synthetic tests isolate Rules 2 / 3 on zones 20 pips apart — narrow by the new cap too, so under
    the default they would now fire 2, not 5 / 3 (the user's AND-ed rule working). They pass `min_proximity_reference_
    gap_pips=_NO_REF_CAP` (documented); `test_mid_cycle_crossing_lifts_caps` also pins the default-cap outcome
    `[5, 10]` (reference narrow + threshold wide → the cap binds).
  - The parent BOS / CTS rungs carry their real as-of outer though the pass test never reads it — so the mutant "the
    last rung tested" is behavioural, not a crash.
  - Mutant 1-14 (banker's rounding) first SURVIVED: test 7's 40.5 case computes as 40.500000000001 in float, so every
    rounding gives 41. Fixed by a pair whose half pip float lands just below .5 — `(0.57405 − 0.57) / 0.0001 =
    40.49999999999998` → 41 (the 1e-9 is for this); banker's and a floor without the epsilon (1-22) give 40.
  Prediction met: vs S_Q 0 / 27 CSVs, 3 / 3 figures equal (hovers included), counts 81/236, 146/135, 289/293, var3 = 2 /
  var4 = 1, pool 8 / 11 / 4, 0 `sibling-CTS unavailable`, skip grep 22, the diag's per-cycle counts 1 / 2 / 3. The §1
  simulator (`build/m4_run.py`, log `build/m4_run.log`) gives the same step-1 trace (passes 754 / 1047; (1,0) narrow
  30 from its start, nothing reaches). Suite 1778 (+26: `tests/test_proximity_reference_ladders.py` — §4's tests 1–14
  + the outer-as-of test + the protocol test; the embedded S_Q rows `tests/_plan_r_reference_window.py`). Mutants
  22 / 22 KILLED (`build/mutants_1.py`: §4's ten + 12 extra incl. the four rename sites). Replay 48.7 s.
- **The combined landing review of 0a + 0b + 1 — DONE (2 read-only agents, 0.51M tokens: code vs plan 0.28M, tests +
  mutants 0.24M; stated 0.4–0.6M).** Code: step 1 matches §1 / §4 and `m4_sim`'s step-1 path (a 4000-case fuzz of
  `CycleProximityScan` + `set_overlay` vs `m4_sim`'s loop: 0 mismatches; a 3000-seed fuzz of the batch vs `bd3bf66`
  with the cap off: every difference preceded by a POI pass). Findings and what was done (the follow-up commit after
  `d26208e`, zero change: 0 / 27 CSVs, 3 / 3 figures, suite 1786):
  - **0a's bound REVISED** (§2, the user: "Establishment moment"): `m15_parent_cts_established_idx` = the LOH of the
    parent cycle's CTS-established moment, not the real-time floor (which empties the window of a trigger fired
    before it — a retroactive cycle with a real-time lifecycle); `lo <= hi` asserted.
  - MARKET_STRUCTURE_SPEC's "Trigger threshold per timeframe" table read 9 / 6 / 3 and claimed a per-TF fallback in
    `MarketStructure` — fixed (8 / 5 / 3; the H1 fallbacks; the probe's H1-pips-on-M15).
  - One class, one name: "CTS-prox class" → "opp_sd-prox class" (LANDMINES, WVMI_SPEC, PART4 §8.5, GOTCHAS, the
    register, a test docstring); the cap's wording "one trigger per side, until its own outward reset";
    `CycleProximityScan` keeps `proximity_pips` raw (no `int()`); the sims README notes `r5_shadow.py fix0a`'s
    pre-0a signature.
  - The test gaps (7 survivors of the reviewer's 9 new mutants — all on step-2 paths: the opp_sd / overlay pass test,
    an opp_sd outward reset while narrow, the sd = −1 cap, the outer at the pass site, the gap-None path; N7 the
    batch order): 7 tests added (`test_review_1…7`) + the batch-order assert in test 10. `build/mutants_review.py`
    (N1–N10 + the 0a-bound R0a-1…3): 13 / 13 KILLED. Every results file saved beside its mutant file.
  - Carried to step 2 (the reviewer's heads-ups): the opp_sd unlock (Q2 / Q3) needs new state; `_kl_outer_at` is the
    parent's H1 clock — sub zones need LOH + the frozen end cap; the first-sd == WVMI-gate assert lives in step 2.
