# Session-order item 4 (2026-10-20) — the IC identification revisit: findings

The user's write-up (verbatim): `user_writeup_20261020.md` — Step 1B (prefer IC candidates whose closest prior
imbalance is unfilled as of the IC) + the imbalance fill threshold 70 % → 65 %. Baseline S_R3
`20261020_225749_d424099` (HEAD `896db93`, suite 1839). Indices are entity-absolute (M15 subs: the lens CSVs'); times
UTC. Round 1 = the review sent to the user (2026-10-20); round 2 = the user's answers (verbatim) + their checks
(2026-10-21); round 3 = the fill-rule decision (separate candles, engine-wide) + its variant replay (2026-10-21).
- `make_variant_stroke2_later.py DIR` — patches a copy (after `make_variant_fill65.py`) so stroke 2 must come on a
  later candle than stroke 1, in all three implementations; the variant replay ran in `%TEMP%/vS`.
- `shadow_fill_rule.py OUT_LOG` — read-only (~20–40 s): the fill-rule census (same-candle commits, closes AT the edge)
  and Step 1B (the user's round-2 flow) under the fill-rule variants E / S / B / SB / EN. Log: `shadow_fill_rule.log`.

Tools (run from the repo root):
- `shadow_step1b.py OUT_LOG` — read-only, no engine edit (~18 s, fetch gate 4228 / 5): every IC record's candidates
  with their closest prior imbalance under each reading, the pick under each flow, at 0.70 and 0.65 (IC side only),
  MS's in-flight reads whose inner would differ, and candle tables for the examples. Log: `shadow_step1b.log`.
- `make_variant_fill65.py DIR` — patches a `git archive HEAD` copy so every production fill threshold is 0.65 (6
  defaults in 5 files); the variant replay ran in `%TEMP%/v65` (59.3 s wall, fetch gate 4228 / 5, skip grep 23).

## The 65 % change alone (every reader) — the variant replay vs S_R3

The threshold is stroke 1 of the two-stroke fill (`ImbalanceInstance.is_filled`: a candle retraces >= thr into the gap
= armed; then a close back past the gap's outer edge = committed-filled). ONE value, held today in two pinned-equal
constants + four defaults: `FIB_FILL_THRESHOLD` (`zones/fib_tracker.py` — the downstream FibTracker AND MS's embedded
one: fib activation / deactivation, the cross-cycle own test + dead-cycle walk), `POIConfig.fill_threshold`
(`zones/poi_zones.py` — IC condition 3, the most-recent-before-imbalance rule's reference, the POI activation sweep's
fill cache, MS's in-flight IC read via `poi_inner_for_fib`), the defaults of `ImbalanceInstance.is_filled`,
`has_unfilled_imbalance` / `iter_unfilled_imbalances`, `resolve_cross_cycle_eligibility`. Pin:
`tests/test_plan_i_s5_ms_reads_the_fib.py` (FIB == FibTrackerConfig == POIConfig == 0.70).

`cmp_save.py` S_R3 vs the variant: **11 real cells, all three figures identical** (traces / shapes 81/236, 153/138,
301/309, xy equal); 0 structure-event, POI, KL, trigger, sub or WVMI changes:
- H1 `imbalance_instances.csv` meta: 7 instances arm earlier (275 → 267, 445 → 444, 494 → 493, 510 → 509, 868 → 866,
  1042 → 1041; instance #92 armed + committed 761 → 600);
- sub 3 cycle 0's fib (both lenses' `fib_lifecycle.csv` meta): one more deactivate / reactivate (last pair 2667 / 2672
  → 2722 / 2723 — the meta keeps only the last pair);
- one IC check (both lenses' `ic_candidates.csv`): its immediate gap 3488 is filled at 0.65 → fails IC condition 3
  (not a pick before or after).

## Step 1B — the shadow (27 drawn records: H1 main 5 + subs 22; + 24 capped sub views of the proximity overlay)

Primary reading P = sd gaps only, the gap's c2 run at or after the fib's BOS anchor and ending before the IC (its c3
may BE the IC), "unfilled" = the engine's two-stroke test with the fills counted THROUGH the IC candle.

At 0.70, P, the write-up's flow (F1: a lone passer is final; none → all candidates): **6 drawn picks move, no-IC 7 → 4**:
- three no-IC records gain a PINBAR IC with NO pattern (each the lone passer): H1 sid 0 cycle 0 → 183, H1 sid 1 cross
  0→1 → 739, counter/confluence sub 3 cycle 0 → 2721;
- sub 2 cycle 1: 2575 (maru, ranked) → 2578 (pinbar, no pattern) — 2578's closest prior gap 2577 is 0.3 pip and its
  c3 IS 2578; 2575's (2564, 1.1 pip) committed-filled at 2566;
- sub 0 cycle 2: 1216 (maru, ranked) → 1142 (normal; 1216's gap 1205, 0.6 pip, filled at 1207; 1142's gap 1140 open);
- sub 3 cross 1→2: 2930 (pattern 3C.iv) → 2929 (normal, no pattern): 2930's own upper wick arms gap 2926 and its
  close (0.58107 = the gap bottom) commits it — 2929, one candle earlier in the same pullback, passes.
MS in-flight reads whose inner would differ (first order): H1 3 / 56, M15 23 / 398.

F4 (a lone passer still goes through Steps 2–3; no IC from the passers → Steps 2–3 on ALL candidates): **2 drawn moves
(sub 0 c2 → 1142, sub 3 cross 1→2 → 2929), no-IC stays 7**; MS reads H1 0 / 56, M15 19 / 398.

At 0.65 (IC side only, today's fibs), additionally: sub 3 cycle 1 2808 (pattern 3B.iii) → 2821 (its gap 2759..2760
commits at 2795 < 2808); sub 3 cycle 0's 2721 no longer passes (2721 itself arms + commits gap 2709). Today's flow at
0.65 moves no pick.

Other readings (F1, 0.70): ARM (stroke 1 alone = filled) adds H1 sid 1 c2 859 → 865 (pinbar), sub 3 c1 2808 → 2821,
sub 5 3427 → 3442; EXC (as of the IC's open) keeps 2930 and moves sub 2 c0 none → 2406 (2406 fills its own prior gap);
LEG (as of the end of the IC's pullback run) fails both 2929 and 2930; C3 (the gap must be complete before the IC)
keeps sub 2 c1 at 2575; ALL (no BOS-anchor bound) moves no drawn pick vs P; ANY (either direction) moves sub 1 c2 →
1935. Step 1A's candidates sit near the BOS anchor — 2 of the 4 Step 1A records have no candidate with a prior sd gap
inside the window (sub 0 c1, sub 1 c2); a BOS-anchor candidate never has one.

## Round 2 (2026-10-21) — the user's answers to round 1, VERBATIM

> Yes the 70% to 65% change is only on the armed portion of the filled imbalance check. And yes the scope is everywhere.
>
> Ahh yes that is a good callout on pinbar. In that case, we may need to make the logic more complex for loner IC after 1B.
>
> If we reach 1B after Step 1 and get loner:
> - loner goes through Steps 2-3
> - if fail Step 2 AND is pinbar, then we return to run Step 1 candidates through Step 2-3
>
> If we reach 1B after Step 1A and get loner:
> - loner goes through Steps 2-3
> - if fail Step 2 AND is pinbar, then we return to run Step 1 candidates through Step 2-3
>
> Actually now thinking more about, it I don't think we need to implement this specifically for loners. In fact, it should be for all cases when we reach 1B runs and we don't produce a final IC (either fail 1B immediately, or fail Steps 2-3 after any number of candidates pass 1B). In that case, we always go back and run either the original Step 1 or original Step 1A candidates (whichever came before the failed 1B) through Steps 2-3.
>
> Does that make sense?
>
> Q2. yes IC candle counts. However, if say there was immediate imbalance prior and the IC comes in to filled the gap, because the imbalance filled definition requires armed and filled, the IC can never actually fill the gap, but only arm it since it is an opposite direction candle.
>
> Q3. Same as above, filled requires armed and another candle to close beyond
>
> Q4. Yes you are correct it should be in direction of structure & opposite of IC. Yes BOS anchor to IC.
> A gap formed by IC counts. IC itself can never be a gap candle because it is opposite direction, but it can be part of the pattern with a prior gap candle.
> Yes no min size
>
> Q5. yes
>
> Everything else sounds good.
>
> Are there anything else you'd like to clarify or discuss?

Decided by these answers: 65 % = stroke 1 only, every reader; 1B scope = sd gaps (opposite the IC), BOS anchor → IC, a
gap whose c3 is the IC counts, no minimum size; 1B outranks the institutional pattern (round-1 Q5).

## Round 2 — checking the answers' assumptions (`shadow_fill_rule.py`, read-only, log `shadow_fill_rule.log`)

**The flow = the shadow's F4** (any number of passers → Steps 2–3; no IC from them → Steps 2–3 on the candidates of
the step before 1B; no passers → Steps 2–3 on all). Consequence: the passers give no IC only when every passer is a
pinbar with no pattern, and the fallback is exactly today's selection — so **1B never adds or removes an IC; it can
only swap today's IC for a passer that wins Steps 2–3 among the passers** (no-IC stays 7 in every variant below).

**"The IC can never fill the gap, only arm it" / "filled requires armed and ANOTHER candle to close beyond" — NOT the
engine's rule.** `ImbalanceInstance.is_filled` (IMBALANCE_FILL_SEMANTICS "The predicate", since `6069ade`
2026-05-23, pinned by `test_imbalance.py`'s same-candle case): stroke 2 = the first candle in `[armed_idx,
check_to_idx]` closing AT or past the outer edge (`<=` bottom / `>=` top) — the arming candle itself can commit. An
opposite-direction candle does both when its body stays outside the gap and its wick reaches >= 70 % in (a pinbar
shape). Round 1's reply described stroke 2 as "a later candle closes back past the gap's outer edge" — wrong on both
counts (same candle allowed; a close AT the edge counts); corrected in round 2.
- Census at 0.70 (every imbalance instance; M15 deduped entity-absolute): H1 167 instances, 152 committed, **48
  committed on the arming candle**, 4 by a close exactly AT the edge; M15 552 / 522 / **153** / 34 (0.65: H1 47 / 4,
  M15 158 / 35). The spec's "rare but legal" is wrong (~30 %) — a doc fix.
- Opposite-direction examples: H1 gap c2 40 (sell, [0.56608, 0.56636]) — candle 42, a BUY pinbar, o .56536 h .56635
  l .56472 c .56594: wick to .56635 arms, close .56594 below the bottom commits; sub 3 2930 (buy pinbar, close .58107
  = gap 2926's bottom .58107 exactly).
- The user's Q4 statement holds: the IC can never be a gap's c2 (`patterns/imbalance.py` requires c2's direction ==
  the gap's; the IC's agreed direction is −sd).
- If "another candle" were adopted GLOBALLY (every fill reader), the commit candle moves for 48 H1 / 153 M15 instances
  (1 H1 never commits); "strictly beyond" moves 4 / 34; both 50 / 179 — a separate cause (its own variant replay).

**1B under each fill rule** (F4, scope as decided, fills counted through the IC; drawn records; 0.65 = IC side only):

| fill rule | 0.70 moves | 0.65 moves |
|---|---|---|
| E — the engine today | sub 0 c2 1216 → 1142; sub 3 cross 1→2 2930 → 2929 | + sub 3 c1 2808 → 2821 |
| S — stroke 2 on a LATER candle | sub 0 c2 → 1142 (2930 kept) | sub 0 c2 → 1142; sub 3 c1 → 2821 |
| B — a close strictly PAST the edge | sub 0 c2 → 1138 (2930 kept) | sub 0 c2 → 1138; sub 3 c1 → 2821 |
| S + B | as B | as B |
| X — E unchanged; 1B reads the gap as of the candle BEFORE the IC (a gap whose c3 is the IC still counts) | as S | as S |

X (proposed to the user in round 2) keeps the engine's one fill rule and implements "the IC can only arm the gap":
2930, 2406, 2721 pass (their gaps are not even armed before the IC). X vs S differ only when an earlier candle arms the
closest prior gap and the IC's close completes it (S: filled; X: not) — 0 cases on the window.

## Round 3 (2026-10-21) — the user's answers to round 2, VERBATIM

> 2. IC can fill the gap by itself.
> If that is the case, then we may need to change the overall logic of filled. Given that there are 2 steps to an imbalance filled, it seems like currently we don't have a rule that specifies that the same candle cannot meet both rules. That means whatever candle meets the armed condition, the 2nd condition for filled must be met by a different candle after the armed candle. Thus, even though the IC is the furthest candle of the pullback leg, even if it does arm the first part of the imbalance filled logic, that candle by itself can never fulfill the filled condition. This makes it so that it is ok to check up to the IC because the IC itself can never fill an imbalance if we make the 2 step must be each met by separate candles.
>
> 3. On edge is ok
>
> Anything else you need from me? Or anything else we should discuss and clarify so we don't miss any details or nuances?

**Decided:** round 2's option (c) — the fill rule changes ENGINE-WIDE: stroke 2 must come on a candle strictly AFTER
the arming candle (every fill reader); a close exactly AT the outer edge still counts; 1B judges the closest prior gap
with fills counted THROUGH the IC. Three changes, in order: 65 % (stroke 1) → separate candles (stroke 2) → Step 1B.

## Round 3 — checks (variant replay `make_variant_stroke2_later.py`, on top of `make_variant_fill65.py`)

The rule lives in THREE implementations: `ImbalanceInstance.is_filled`'s positional fast path and its label-based
fallback (`common/types.py`), and `_compute_fill_idx_cache` (`zones/poi_zones.py`, the POI activation sweep's numpy
scan, "same-candle fill is legal" comment). Pinned today by `tests/test_imbalance.py`'s same-candle case (it flips).

Variant replay `%TEMP%/vS` (HEAD + 65 % + separate candles; 56.2 s wall: data_fetch 6.24, multi_tf_dual 11.09,
chart_h1 6.15, chart_m15_counter 9.30, chart_m15_confluence 22.27; fetch gate 4228 / 5; skip grep 23), `cmp_save.py`
vs the 65 % variant's outputs (`%TEMP%/b65`): **51 real cells; 0 structure-event, fib, IC, KL, trigger, sub or WVMI
changes**:
- H1 `imbalance_instances.csv` meta: 47 instances' `confirmed_fill_idx` move later (e.g. 42 → 47, 106 → 110, 954 →
  964); #35 (gap c2 180, armed 183) never commits (= the census at 0.65: 47 / 1);
- H1 POI sid 1 cycle 2 (IC 859): the 992 deactivate / 998 reactivate pair disappears (gap 987's fill moves 992 → 999,
  when gap 997 already keeps it active) → `confirmed_idx` 998 → 954; drawn as ONE continuous zone 2026-01-13 16:00 →
  01-19 23:00 instead of two segments + a reactivation line — on all three charts (shapes 236 → 234, 138 → 136,
  309 → 307; traces xy-equal);
- M15 confluence: two sub POIs deactivate one candle later (sub 1 cycle 2: 2192 → 2193; sub 4 cycle 0: 3377 → 3378).

**The user's premise "the IC itself can never fill an imbalance" holds only when the IC is the ARMING candle.** Under
separate candles the IC can still be the SECOND candle: an earlier pullback candle arms the gap and the IC's close
completes it. On H1 (65 % + separate candles) 3 fills are completed by a candle pointing against the gap (the IC's
direction), e.g. gap c2 447 (buy, [0.58154, 0.58162]): armed by 449 (12-11 15:00, buy normal, low .58144), committed
by 450 (12-11 16:00, SELL pinbar, o .58218 c .58182 >= the top). Among the window's IC candidates: 0 such cases (1B's
shadow S column). Through the IC (the user's rule), such an IC fails 1B — consistent with the definition (two candles,
both by the IC's close).

1B under the decided rules (first order, drawn records; the 1B shadow's S variant at 0.65 — valid since neither fill
change moves a fib, IC candidate check or pick): **sub 0 cycle 2 1216 → 1142; sub 3 cycle 1 2808 → 2821**; sub 3
cross 1→2 keeps 2930; no-IC stays 7. MS's in-flight cascade: to measure at plan time.

## Round 4 (2026-10-21) — the user's answer to round 3, VERBATIM

> Separate Candle Rule.
>
> "But it can still complete a fill that an earlier pullback candle armed."
> Your logic on the example here doesn't really make sense because 450 can never be an IC candle. 450 is in the same direction as the imbalance and in sd, and IC must always be opposite direction. If 450 here was in opposite direction, it would further retrace and not be what's needed to fill the imbalance and the 2nd step.
>
> The only scenario where something like that could happen would be if 450 was both opp in candle direction but sd in volume direction. This would cause the candle to be red, but it to lie above 449 potentially. I'm not sure what logic we currently have around IC direction, but if we don't already we should also require IC to be both opposite in candle direction and opposite in volume direction of structure direction.
>
> Anything else worth discussing or deciding? Want to make a last pass

## Round 4 — checks

**450 IS an opposite candle in both senses** (H1 final CSV, S_R3): o .58218 c .58182 → candle direction −1 (red);
`vol_dir` −1 (close below 449's .58218); agreed −1 — it passes IC condition 1 in a buy structure (struct_direction
+1 there). It retraces, yet closes ABOVE gap 447's top (.58162): 449 had already recovered to .58218, so 450's lower
close is still outside the gap. Stroke 2 asks WHERE the close is (at / past the outer edge), not which way the candle
moved — so under separate candles the fill typically completes on the next candle closing outside the gap, whatever
its colour, and an IC can be that second candle. (450 itself is in no fib window; 0 such cases among the window's IC
candidates.) The user's "red but above 449" case = a SPLIT candle (candle −1, volume +1) — already excluded by
condition 1 today.

**IC condition 1 already requires both directions** — `agreed_direction(direction, vol_dir) == −sd` (GLOSSARY "Agreed
direction", PLAN_H §1 item 2): split candles are excluded. Two relaxations pass today: VOL0 (candle −sd, close ==
previous close) and DOJI (open == close, volume −sd). `shadow_ic_direction.py` (log `shadow_ic_direction.log`; drawn
records; condition 1's row set only — the pattern legs keep the agreed array):
- Step 1 rows: 1651 both / 32 VOL0 / 31 DOJI; IC candidates 41 / 1 (H1 sid 1 c2 865, pinbar, not picked) / 1; picks
  19 / 0 / **1 — sub 4 cycle 0's IC 3235** (confluence lens; pattern 3A.iv, legs 3232..3234 | 3235 | 3236..3238): a
  doji pinbar, o = c = .57644, its close 0.1 pip below 3234's .57645.
- STRICT (both dropped, the proposal literally) = NODOJI: sub 4 cycle 0 → no IC (Step 1A, all pinbar); MS reads whose
  inner differs: H1 0 / 56, M15 7 / 398. NOVOL0 alone: no pick moves.

## Round 5 (2026-10-21) — the user's decisions, VERBATIM

> Let's go with a) for condition 1. Also I've changed my mind, I now want to exclude IC from being able to fill most recent prior imbalance. Basically the balance needs to be unfilled up to the candle right before the IC. Anything else to bring up? If not we can proceed

**Decided:** IC condition 1 unchanged (the agreed direction; flat-volume and volume-only-doji candles still qualify).
Step 1B reads the closest prior gap's fill state through the candle RIGHT BEFORE the IC (the shadow's X; with the
separate-candles rule = SX) — the IC can never disqualify itself. A gap whose c3 IS the IC still counts (round 2 Q4):
its fill scan starts at the IC, so through IC − 1 it is unfilled by construction.

Measured (`shadow_fill_rule.py` SX, F4 flow, drawn records; at 0.65 IC side only — neither fill change moves a fib,
IC check or pick): **0.65: sub 0 cycle 2 1216 → 1142, sub 3 cycle 1 2808 → 2821**; 2930, 2406, 2721 pass (their
gaps are not armed before the IC; 2406 / 2721 are lone no-pattern pinbars → the fallback keeps today's no IC);
no-IC stays 7. X == SX on every candidate here.

**Final spec for the plan** (rounds 1–5): (1) stroke 1 at 0.65, every reader, one constant; (2) stroke 2 on a candle
strictly after the arming candle, a close AT the edge counts, every reader (three implementations); (3) Step 1B after
Step 1 / Step 1A: passers = candidates whose closest prior sd gap (c2 run from the BOS anchor, ending before the IC;
"closest" = ends latest; merged instances; no minimum size) is unfilled through IC − 1; passers → Steps 2–3; no IC
from them → Steps 2–3 on all candidates of the step before (= today's pick); no passers → Steps 2–3 on all; 1B
outranks the institutional pattern; no prior gap → fails 1B; a gap straddling the BOS anchor does not count (0 on the
window); Step 4 deferred. Condition 1 unchanged.

- 2930 fails only under E (it arms gap 2926 and closes exactly on its bottom). 1138 (sub 0 c2): its gap 1133 (buy,
  [0.57270, 0.57288]) armed by 1136 (low .57271), committed by 1137's close .57288 = the top exactly — another candle,
  AT the edge → fails under E and S, passes under B; with 1138 and 1142 both passing, the ranking's oldest-candidate
  rule picks 1138.
- No closest-prior gap committed BY the IC after an earlier candle armed it (the S-vs-"fills through IC − 1" case): 0.
  The closest prior gap committed ON the IC: 2930, 2406 (0.70), + 2721 (0.65) — all armed by the IC too.
- A candidate with NO prior sd gap in the window: 5 of 43 (922, 923, 1934, 1935, 4027). Counting it as a passer (EN)
  moves no pick; default: it fails (no imbalance is not "an unfilled imbalance").
