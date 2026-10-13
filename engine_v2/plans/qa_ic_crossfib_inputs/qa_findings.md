# Session 2 (2026-10-12) — the IC identification + cross-cycle fib Q&A: findings

The user's questions on S_N_hover (`artifacts/commits/week9-entries-decisions/20261012_194022_9f9406d`), answered from
the code + an instrumented replay (no engine edit). Explanations only — no decision taken, nothing changed. Indices are
M15 entity-absolute (the counter lens CSVs); times UTC.

Tools (run from the repo root; each runs the replay pipeline in-process, no exports, ~20–90 s; fetch gate 4228 / 5):
- `trace_sub_cross_fib.py SUB_ID` — every handled CTS event of the sub's DOWNSTREAM FibTracker (the drawn fib): the
  limit, phase, the own-window gaps and the dead-cycle walk per cycle with each gap's armed / filled candle.
- `trace_sub_ic_inputs.py SUB_ID CYCLES LO HI` — `select_ic` of the drawn records: zone bounds, the IC reference, the
  candidates' patterns, condition 3's gaps, the candle rows (type, candle / volume / agreed direction).
Replica check: both reproduce the saved fib records (anchors, versions, `via`) and IC picks exactly.

## Q1 — sub 3 (counter, sd −1): why only the 1→2 cross

- Limit: `P` 0.5736 (`parent_bos_threshold`, no move); `M` = 3 — cycle 3's CTS anchor (0.57318 at 3589) is the first
  strictly below `P` (cycles 0–2: 0.5818 / 0.57902 / 0.574 — cycle 2's 0.4 pip short). Cycles 1, 2, 3 all ALLOWED (fixed
  True at 2829 / 2992 / 3589). The limit blocked nothing.
- The walk (CROSS_CYCLE_FIB_SPEC §4): back from T−1, stop at the first DEAD cycle (no unfilled bearish gap in its own
  [BOS_k, CTS_k], fills counted to the check's moment; dead is cached forever) — a cross never skips a dead cycle.
  - cycle 0 [2639, 2736]: 8 gaps, all filled by 2760 → DEAD at the first check, 2816 (the first new low past CTS_0
    0.5818 after its confirmation 2752; at 2752 the gaps 2709 / 2722 / 2724 were still unfilled).
  - cycle 1 [2758, 2842]: LIVE — gap 2810..2812 [0.58227, 0.58296] never armed (needs a high ≥ 0.58275).
  - cycle 2 [2915, 3047]: 18 gaps, all filled by 3308 (2931 → 3300, 2962..2963 → 3306, 2966..2967 → 3308) → DEAD at
    the first check for cycle 3, 3521 (the first new low past CTS_2 0.574, 435 candles after its confirmation 3086).
- Result: 0→1 ✗ (cycle 0 dead) → cycle 1 single `cross_failed`; 1→2 ✓ (created 2967, the first new low past CTS_1),
  0→2 ✗ (stops at dead cycle 0); 2→3 ✗ (cycle 2 dead) → cycle 3 single `cross_failed`; 1→3 / 0→3 ✗ (the walk stops at
  dead cycle 2 before it reaches live cycle 1). Own-window test passed at every check.

## Q2 — cycle 2 cross 2758→3047, IC 2821 (`no_pattern`, no ranking rule)

- Zone [0.580328, 0.582192]; IC reference 0.5818 `prev_cycle_cts` (cycle 0's CTS; bound 2824 = cycle 1's establishing
  pattern) beats `cross_limit_anchor` 0.5736 (sell → the higher). Condition 2 = low > 0.5818 and idx < 2824.
- Step 1: 130 agreed-buy candles; touch 55, reference 26, unfilled-after 129; ALL = 2821 only. The touch needs low ≤
  0.582192, the reference low > 0.5818 → a 3.9-pip band before 2824: 2762–2808 sit above the zone (lows ≥ 0.58249),
  2817 / 2819 (lows 0.58166) fail the reference, 2821 (low 0.58182) passes by 0.2 pip.
- Step 2: 2821's only variant — initial 2820 (pinbar, 0.6 pip body) | opp 2821 maru | continuation 2822–2824 (big
  maru 3.93, body 13.7 > 3.9) → `initial_fails` (an initial must be big maru / maru / big normal; 2819 is a buy candle,
  so the initial is 2820 alone). No pattern.
- Step 3: rule 1 keeps 2821 (maru); exactly one IC candidate, not a pinbar → picked with `ranking_rule` EMPTY (ICR-10).
- Debug-only patterns in the window (not IC candidates — all fail the reference): 2817 2A.ii, 2837 1A.iv, 2919 2A.iv,
  2930 3C.iv.

## Q3 — cycle 1 single 2758→2842, IC 2798 (2B.iii)

- Zone [0.582246, 0.583196]; the same reference (0.5818, before 2824).
- Step 1: 34 checked; 26 IC candidates (2762–2808 + 2821); 8 fail both the touch and the reference (2817, 2819, 2825,
  2826, 2830, 2831, 2832, 2837).
- Step 2: 2798 2B.iii (2797 big normal 0.69 | 2798 maru | 2799–2801 big maru 0.90, 4.6 > 2.7) and 2808 3B.iii (2807
  big normal 0.81 | 2808 pinbar | 2809–2811 big maru 1.53, 6.3 > 1.4); 2B beats 3B → 2798, no tie, Step 3 not run.
  The other 24: `ic_not_last_opp` 10, `continuation_not_big_maru` 9, `no_initial_candle` 2 (split candles 2784 /
  2791 before the opp leg), `no_continuation` 1 (2783: split 2784 next), `initial_fails_multi_opp` 1 (2788),
  `initial_fails` 1 (2821).

## 2026-10-13 — the user's three-change write-up (`user_writeup_20261013.md`): the read-only shadow

`shadow_ic_changes.py` (no engine edit; 58 s; fetch gate 4228 / 5): every drawn POI record (26 — H1 main 5, subs 21),
the pick today (V0, asserted == the real pick) vs change 1 (V1: condition 3 = the immediate unfilled imbalance), change
2 (V2: the CTS reference = CTS_{T−1} for a cross too), 1+2 (V12), 1+2+3 (V123: no candidate in 61.8–80 → 80–100).
FIRST-ORDER only (today's structure); MS reads the same IC (sd-proximity), so the real replay can cascade.

- No-IC records: V0 1 · V1 12 · V2 1 · V12 11 · V123 7. Picks moved: V1 19 · V2 1 · V12 19 · V123 20.
- MS in-flight reads whose POI inner would differ (H1 / M15): V1 51 of 56 / 338 of 429 · V2 0 / 19 · V123 51 / 342.
- The user's predictions: change 1 moves sub 3 cycle 1's IC ✓ (2798 2B.iii → 2808 3B.iii, 12-29 04:00, pinbar;
  candidates 2808 [gap c2 2810] + 2821 [gap 2823..2824, unfilled at 2842]); change 2 moves sub 3's 1→2 cross IC ✓
  (alone: 2821 → 2837 1A.iv, 12-29 11:15 maru, 55 candidates 2817…2961; with change 1: 2930 3C.iv, 12-30 10:30
  pinbar, candidates 2929 / 2930 [gap 2931] + 2960 / 2961 [gap 2962..2963]). Change 1 ALONE leaves that cross with
  NO IC (2821's gap 2823..2824 filled at 2917 < CTS 3047).
- Change 2 alone moves only that record (4 crosses in the window; singles are unchanged by construction: a = T).
- Change 3 rescues 4 (sub0 c0 → 650, sub0 c1 → 923 3C.iv, sub1 c2 → 1934 [no IC today either], sub7 cross 0→2 → 4131
  3B.iii). The 7 left: all-pinbar in 61.8–80 (H1 sid0 c0, H1 sid1 c0, H1 sid1 cross 0→1, sub3 c0 — Step 1A does not
  run by design), all-pinbar in 80–100 (sub2 c0), no candidate in either (sub1 c0, sub1 c3).
- Review points raised (2026-10-13, for the user's decision): (a) with change 1 the last rule of BOTH steps always
  picks the OLDEST candidate in play (its own immediate gap is the earliest after it; Step 3: or the neighbour sharing
  that gap) — Step 2 already did when all ties passed the immediate rule (TBU-4); 0 of 26 records reach it under
  V1/V12/V123; (b) change 2 on a cross that exists before its target cycle is established (created on the first new
  extreme past CTS_{T−1}): no break pattern yet → price side only (RX1-2a's convention) — today unreachable (a < T is
  always established); `stamp_ic_references` + MS's read can see it, the drawn POI cannot [CORRECTED 2026-10-13,
  PLAN_O §2 + its plan review R6: ONLY the stamp can (at the lifecycle cap, a cross still pre-established at its
  structure's end / the data edge); the drawn POI never (no record for a cycle with no CTS_ESTABLISHED,
  `poi_zones.py:820-830`); MS's read never (it refreshes only at its current, established cycle's CTS_ESTABLISHED /
  CTS_UPDATED); 0 such reads on the window, `shadow_change2.py`]; (c) change 3's coupling:
  the POI activation sweep re-checks the IC's touch of the 61.8–80 zone as of each candle (condition 5,
  `ic_not_touching_fib_zone`) — a 80–100 IC needs its own band there.

## 2026-10-13 — the user's decisions on the review points (verbatim)

1. (with change 1, both steps' last rule picks the OLDEST candidate in play) — "Yes"
2. (change 2 before the target cycle is established: price side only until the break pattern exists) — "Ok"
3. (change 3: a 80–100 IC re-checks 80–100 in the activation) — "Yes, POI activation check should always reference
   the zone parameters the IC came from"
4. (7 of 26 fibs with no POI after all three, until Step 4) — "For now it is. It's difficult to judge whether that
   change is truly impactful until I see the changes and the chart outputs."
5. (order: change 2 → change 1 → change 3, each its own plan + /compare) — "Sounds good on your order."

The user's alternative, asked for a take (not chosen): all three at once, H1 structure only, "less chart elements and
may be easier to get a sense of the impact before moving onto the subs". The take (2026-10-13): not recommended —
first-order on H1 only change 1 shows (H1: change 2 moves nothing, change 3 rescues nothing — its 3 losses are
all-pinbar), the subs move anyway (H1 POIs feed the zone-proximity triggers; H1 cycles feed sub geometry / `P`), one
logic for main + sub would need a temporary TF gate in `select_ic` + MS's read + the activation, and three causes in
one /compare. Offered instead: the H1 chart first in each chart review; optionally an uncommitted all-three TRIAL
replay (look only, then land one at a time).

## 2026-10-13 — the combined TRIAL replay (the user: "I do want the combined preview first")

All three changes built UNCOMMITTED in the working tree (patches kept here, cumulative: `trial_p1_change2.patch`,
`trial_p2_change2_1.patch`, `trial_p3_all.patch` — the landings' starting points, NOT landings: no tests / docs /
cleanup). Trial-only shape: `has_unfilled_imbalance_after` kept as information; the immediate tiebreak / ranking rule
left in (a no-op); `ic_zone_band` on `ICSelection` / POI meta / IC CSV (new column after `is_ic_candidate`) / both
hovers ("IC zone 80-100 (fallback)", fib status "· zone 80-100 (fallback)"). Replay 48.7 s, fetch gate PASS
(4228 / 5), skip-grep 22 (= baseline). Keyed diff vs S_N_hover: `cmp_trial.py BASE CUR`.

- Every IC outcome = the shadow's V123 prediction (26 of 26).
- Drawn POIs: H1 5 → 2 (180, 694, 732 gone — all-pinbar; 832 → 859; 607 kept), counter 7 → 6, confluence 19 → 16.
- The cascade (MS): ONE structure event on H1 — sid1 cycle 2 CTS_CONFIRMED (sd_zone_proximity on the POI) 939
  (01-13 01:00) → 935 (01-12 21:00): the POI inner 0.57866 (IC 832) → 0.57786 (IC 859). Knock-ons: H1 KL zones +
  WVMI trigger (939 → 935); the counter lens's parent-(1,2) triggers 4 H1 hours earlier (first_counter 3759 → 3743,
  subsequent_counter 4159 → 4143) → sub 5 [3743, 4143] (was [3759, 4159]), sub 7 starts 4143 (was 4159) — before its
  cycle 1's establishment (4157), so its cycle 0 gets a live stretch: its KL / POI (4048) now show (counter chart
  +9 traces / +12 shapes; confluence +7 / +17 also has sub 0's fallback POIs). ZERO M15 structure-event changes;
  triggers in the confluence lens unchanged; unresolved triggers unchanged (4).
- The fallback zone in use: sub 0 c0 650 + c1 923, sub 1 c2 1934, sub 7 cross 0→2 4131 — 923 / 1934 / 4131 activate
  (the band re-check works); 650 never does, and neither did its baseline POI 670: sub 0's cycle 0 ends (1020,
  next_cycle) at the candle sub 0 goes live (`start_idx` 1020), so its POI gets no live candle (CORRECTED 2026-10-13:
  the first report to the user said all four activate). Sub 1 c0 / c3 and sub 2 c0 ran it and found none (sub 2:
  only a pinbar).

## 2026-10-13 — the user's chart look at the TRIAL + the landing split (verbatim)

"From what I've seen so far, I don't see any issues with the changes we've implemented. The zone shifts are generally
more in line with price would retrace back to. Additionally, I don't perceive any of the now missing POI zones as an
issue. I think we can proceed with the changes. I agree that we should implement change 2 first and commit-save the
output. Change 2 I think we will for sure keep. I would like change 1 & 3 to be together in 1 commit. For one they go
together anyways and 3 is only needed if we have 1. Furthermore, change 1&3 are something we may choose to revert or
update in the future, so having them as 1 commit together makes it easier to revert to down the line."

So: TWO landings — change 2 alone (plan, /compare, commit-save), then changes 1 + 3 in ONE plan / commit / save (the
revert unit). Attribution inside the 1 + 3 /compare stays mechanical: every change-3 effect carries
`ic_zone_band` = 80-100, and H1 has no fallback pick, so the H1 cascade (832 → 859, CTS 939 → 935) is change 1's.

## 2026-10-13 — follow-up: the POI activation's imbalance condition under change 1 — DECIDED: (a) keep today's rule

The user (verbatim): "Thank you on keeping POI active, I'd actually like to go with a) keeping today's rule. We should
decouple the rule from IC identification with POI active. The concept of keeping a POI zone active centers more around
that we currently still have unfilled imbalances in our current cycle. Therefore, keeping the old rule suffices in
checking that. IC identification on the other hand is trying to find the best candidate to build the zone around,
making the immediate imbalance part crucial."

For PLAN_P: the sweep's condition 3 stays as built (any sd imbalance formed by t overlapping (ic_idx, t], not
commit-filled); only its WORDING changes — it is the POI's own "still unfilled imbalance" condition, no longer "IC
condition 3 re-checked" (`_compute_poi_activation_history` docstring, the `derive_poi_zones` comment block,
POI_ZONES_SPEC §4). The touch re-check (condition 5) stays coupled to the IC's zone band (decision 3). Precise scope of
"current cycle": gaps after the IC count (from IC + 1 up to the candle being checked); gaps between the BOS and the
IC do not.

### The measurement behind the decision (option (b) not taken)

The activation sweep (`_compute_poi_activation_history`, condition 3) re-checks the OLD IC condition 3 per candle:
any sd imbalance formed by t overlapping (ic_idx, t], not commit-filled. Under change 1 the IC qualifies on its
IMMEDIATE gap; the sweep's own docstring says it re-evaluates "the IC's identification conditions" — the same
coupling as the zone band (decision 3). Readers: the charts' active stretches + `check_zone_proximity`
(`poi_active_as_of`) on the H1 POIs only (→ the counter / confluence triggers); MS's in-flight read ignores activation.

`shadow_activation_immediate.py` (read-only, on the TRIAL tree, 22.5 s): of 19 drawn POIs, 10 would differ if the
sweep counted only the IC's immediate gap — always the same way: the POI switches off for good at the candle that
gap commit-fills (70% retrace into it, then a close back past it in the impulse direction). H1 (sub indices in the
log are sub-local):
- IC 607 (sid0 c1): gap 609, commit-filled 809 (01-05 15:00). Today active 652 (12-24 02:00) → the cycle's end; with
  the immediate gap: off from 809.
- IC 859 (sid1 c2): gap 861..862, commit-filled 944 (01-13 06:00). Today on 902 (01-09 12:00), off 952, on 954, off
  992, on 998 (01-15 12:00) → the end; with the immediate gap: off from 944.
- Sub 7 c0 IC 4048: its gap commit-filled 01-16 16:00, before the POI's first activation (01-19 01:45) → never active.

## 2026-10-13 — handoff to PLAN_P (changes 1 + 3, ONE commit — the user's split)

Baseline: S_O `20261013_151425_ce5ac9b` (PLAN_O landed — change 2). Starting code: **`trial_changes_1_3_on_S_O.patch`**
(the combined trial minus change 2 — `git apply --check` clean on `305fbaa`; built as diff(caf0f01 + p1, caf0f01 + p3)).
It is the trial, NOT a landing: no tests, no docs, the trial-only shape above. Since the trial included change 2, its
outputs (§ "the combined TRIAL replay") are PLAN_P's expected outputs on S_O — re-measure them (a shadow / a variant
replay on S_O) before writing PLAN_P's prediction; the trial's own artifacts were overwritten by PLAN_O's replay.

What the trial left out — PLAN_P must decide / do:
1. **Remove the immediate rule from both rule sets** (the user's write-up): Step 2's tiebreak branch
   (`TIEBREAK_IMMEDIATE_UNFILLED_IMBALANCE`, `poi_zones.py` ~557) and Step 3's ranking branch
   (`RANKING_IMMEDIATE_UNFILLED_IMBALANCE`, ~637). The tuples' 4th element (`ip_tb_imbalance_first_c2_idx` /
   `ranking_imbalance_first_c2_idx`) STAYS — the most-recent-before-imbalance rule fills it too (~571 / ~651). Pins of
   the immediate rule (PLAN_H §6 TBU tests, PLAN_L L2 tests) flip or go; the user's decision 1 (the last rule picks the
   oldest candidate in play) gets a pin.
2. **The old condition 3** (`_has_unfilled_imbalance_after`, `ICCandleCheck.has_unfilled_imbalance_after`, its IC CSV
   column): keep as information or drop — ASK the user (the activation sweep keeps its OWN imbalance check either
   way, decided 2026-10-13).
3. **The activation sweep's imbalance condition stays** (decided): reword its docstring, the `derive_poi_zones`
   comment block and POI_ZONES_SPEC §4 — the POI's own "unfilled sd imbalance after the IC" condition, not "IC
   condition 3 re-checked". The touch re-check (condition 5) uses the IC's band (`ic_zone_config`, decision 3).
4. **Names — confirm with the user** (Naming Standard): `ic_zone_band` ("61.8-80" / "80-100"), `IC_ZONE_PRIMARY` /
   `IC_ZONE_FALLBACK`, `POIConfig.ic_fallback_fib_min/max`, the hovers "IC zone 80-100 (fallback)" (POI) and
   "· zone 80-100 (fallback)" (fib status).
5. **A save-format boundary** in the `/compare` skill (the IC CSV's `ic_zone_band` column after `is_ic_candidate`, the
   POI meta key; + the dropped column if 2 = drop); the keyed diff `cmp_trial.py BASE CUR` reads it.
6. **Step 1A's scope, as decided:** it runs only when Step 1 finds NO candidate (re-runs Step 1 with condition 4 on
   80–100, conditions 1–3 kept), never after Steps 2–3 — an all-pinbar record does not trigger it (H1 sid 0 c0, H1 sid
   1 c0, H1 sid 1 cross 0→1, sub 3 c0 stay POI-less; Step 4 is the deferred answer, register §D.2).
7. **The cascade to re-measure:** MS reads the same IC (`poi_inner_for_fib`) — the trial moved H1 sid 1 cycle 2's
   CTS_CONFIRMED (sd-proximity) 939 → 935 via its POI inner (IC 832 → 859), hence the counter triggers on parent (1,2)
   (3759 → 3743, 4159 → 4143) and subs 5 / 7; 0 M15 structure-event changes. Docs: POI_ZONES_SPEC §3 (Step 1 condition
   3, Step 1A, §3.4 tiebreaks, §3.5 CSV, §3.6 ranking) + §4, CHARTING_SPEC (hovers), GLOSSARY, PLAN_H §6 TBU / ICR
   pointers, the register.
