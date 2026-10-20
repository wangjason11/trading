# The opp_sd zones proximity trigger — Q&A and measurements (2026-10-14)

The user's write-up (verbatim): `user_writeup_20261014.md`. SESSION ORDER item 3 (register `IDEA_PARKING_LOT.md` §D),
after PLAN_Q. Status: **DISCUSSION — no plan yet, no code changed.** Every number below was read from the S_Q save
`artifacts/commits/week9-entries-decisions/20261014_170919_da8c15e` (H1 final CSV, the H1 chart's trigger trace, the
M15 subs / triggers / structure events / KL / POI CSVs) and from the code at `bd3bf66`. H1 candle h covers M15
4h … 4h+3 in this window (checked: H1 826 and M15 3304 are both 2026-01-06 08:00; LOH(902) = 3611).

## The proposal in one line

opp_sd reference set = the parent CTS zone's inner + the most recent counter-direction sub's most recent cycle's BOS
zone inner + POI inner (all sells under a bullish parent); the closest to the sd side wins (bullish parent: the lowest);
triggers still fire on the H1 candles; the alternation, the thresholds and the downstream detectors are unchanged.
Rename "CTS zone proximity trigger" → "opp_sd zones proximity trigger".

## The user's claims, checked (2026-10-14)

| Claim | Verdict | Evidence (S_Q) |
|---|---|---|
| 748 sd:POI on (0,1)'s pullback starts the first counter sub | holds | counter lens `first_counter`, trigger M15 2995 = LOH(748) → sub_3 |
| the rebound never reaches the parent CTS zone | holds | (0,1) CTS inner 0.58404 needs a high ≥ 0.58324; the highest high 749–901 is 0.58105 (826) |
| sub_3 runs counter to sid_0 at 820 / 871 | holds | sub_3 (bearish, starting M15 2639) is live on BOTH lenses: confluence from 2829 (sub_2 reversed into it), counter from 2995; ends 3611 `parent_end` |
| every opp_sd zone is a sell | holds | parent CTS zone sell; sub_3's BOS zones and POIs all `side = sell`; inner = bottom for each |
| an opp_sd fires on sid_0 near 820 | holds — exactly **820** | from 749 sub_3's most recent cycle = cycle 2 (BOS_CONFIRMED M15 2992 = H1 748; CTS_CONFIRMED 3086 = H1 771): BOS inner 0.58171, POI (IC M15 2930) inner 0.58096 → the POI is closest; 820 high 0.58064 = 3.2 pips; the highest high 749–819 = 0.57968 (out of range). Robust to every timing rule below (all of sub_3 cycle 2's moments precede 820) |
| a second sd follows around 830–833 | holds — **833** | (0,1) sd level = max(BOS inner 0.57494, POI top 0.57712) = 0.57712 → low ≤ 0.57792; lows 821–832 all above; 833 low 0.57718 (0.6 pips) |
| #1: the 820 opp_sd starts a `subsequent_confluence` → a new unique sub | holds in principle | its input = the counter lens's latest CTS anchor in [2995, LOH(820) = 3283] = sub_3 cycle 2's CTS anchor M15 3047 (H1 761, the low); no existing sub has the key (3047, bullish). Exact start only by the build |
| #3: the 833 sd starts a `subsequent_counter` whose sub = today's sub_4 | plausible, NOT guaranteed | its input = the new bullish sub's latest CTS anchor in [3283, LOH(833) = 3335]; the high in that window is M15 3304 (H1 826) = sub_4's starting candle (also (1,2)'s BOS anchor and sub_3 cycle 3's BOS anchor). Depends on the new sub's own CTS events + the probe |
| sub numbers shift | holds | ids are numbered by lifecycle START (`number_subs_by_start`): the new bullish sub (start 3283) → 4, "sub_4" → 5, every later id +1 at least |
| "it also … ends sub_3 … via same_dir_replacement" | **wrong today, right after the change** | today sub_3 ends at 3611 by `parent_end`; sub_4 starts 3621 and is itself ended by `same_dir_replacement` (sub_6, 3819). After the change the 833 `subsequent_counter` replaces sub_3 on the counter lens (same lens, parent cycle, direction — `_phase1_record_start`), and a sub ends at its earliest record end (`_phase4_sub_end`) → sub_3 ends at 3335 instead of 3611; its cycle 3 (BOS_CONFIRMED 3589) leaves its lifecycle |
| "this sub_4 would actually already exist … the later first confluence becomes a retrigger" | **the key repeats, but the pool freezes it** | "sub_4" built at 833 is a counter record under (0,1) → ends at `parent_end` 3611. The (1,2) `first_confluence` (trigger 3611, start 3621) finds the same key; start 3621 > the sub's end 3611 → frozen ZERO-LENGTH (`_phase1_record_start`; pool rules are write-once). So under (1,2) the confluence lens has no live sub 3621 → 3819 (today sub_4), and (1,2)'s `first_counter` (3743) reads its sibling CTS from that lens → falls back → sub_5's start may move. Carrying the sub across would need a NEW pool rule |
| "similar to … the cross cycle limits … a peek into the future" | not a peek, as far as measured | every reference of the 820 trigger is known in real time (sub_3 cycle 2: BOS 748, CTS confirmed 771). The change lets sid_0 react in real time to price structure the H1 engine only recognises retroactively at 902. The look-ahead risks are only the two timing rules (decisions 3 and 4) |

## The cascade past 833 (depends on decision 4)

After 833 the opp_sd slot reopens; reference = parent CTS + the new counter sub ("sub_4") cycle 0: BOS inner 0.58055
(BOS_CONFIRMED M15 3306 = H1 826; never reached — the highest high 834–901 = 0.57963) and POI IC M15 3427, inner
0.57885 (could go live at M15 3430 = H1 857; its fib final only at cycle 0's CTS_CONFIRMED M15 3741 = H1 935, anchor
3621 = H1 905).
- POI counted from its go-live floor → opp_sd at 857 (high 0.57918) [or 858 (0.57815) under the strict hour rule] →
  another `subsequent_confluence` → sd at 858 (low 0.57748) [or 859 (0.57786)] → another `subsequent_counter` → …
- POI counted only once its fib is final (H1 935, after sid_0 ends at 902) → the chain stops at 833.

## Engine conflicts found

1. **Circular dependency.** The orchestrator computes every H1 trigger first (`pipeline/orchestrator.py:422`), derives
   the subsequent triggers from that list (598–623), then builds the M15 pool. The new opp_sd needs subs built from
   EARLIER triggers (748 → sub_3 → 820 → new sub → 833 → …). Options: run the H1 scan inside the lifecycle sweep
   (moment-ordered already); or iterate triggers ↔ pool to a fixed point. Sub geometry runs to the data edge, so each
   sub zone must be clipped to "known at that H1 candle" (`sub_structure_pool.knowable_at_idx`).
2. **H1-only runs + the H1 chart.** The H1 trigger list will depend on the M15 pass; a run without lower TFs gives the
   old CTS-only list; the H1 chart reads `attrs["zone_proximity_triggers"]` from the main pipeline — must be fed the
   final list.
3. **Collapsed cycles unchanged.** (1,1) builds no subs → its 871 opp_sd stays CTS-only, its 820 sd:BOS stays; the chart
   will carry two 820 markers (sid_0 opp_sd, sid_1 sd:BOS).
4. **Labels / units.** `ZoneProximityTrigger.zone_kind` is BOS / CTS / POI; a sub zone needs its sub + cycle on the
   hover, and its `zone_confirmed_idx` is an M15 index on an H1 chart. The rename touches code, PART4 §4.3.4 / §4.3.5,
   both detectors' docstrings, the `opp_sd:CTS` label.
5. **`/compare`.** Sub ids renumber → the diff must key on (starting candle, direction).
6. **Unchanged:** the first sd of every cycle (the H1 WVMI gate `first_sd_prox`, MS's sd-proximity CTS confirmation);
   the narrow-cycle rules (the parent's gap; (0,1) = 117 pips, wide).

## Decisions — round 1 (asked 2026-10-14; the user's answers verbatim)

On the corrections: (1) sub_4 doesn't end sub_3 today — "Yes you are correct here. I misspoke earlier". (2) the
zero-length retrigger — "I do not mean that sub_4 would continue or restart. I'm stating that sub_4 when it gets
retriggered later at sid_1, would essentially become one of those retriggers that we never rebuild because all the
results would be the same and we would end up with the same outcomes. So that is equivalent of it already existed and
ended, and a retrigger does not necessitate building the MS again." (3) the cascade — "It should be only after it is
final. What's the condition for a fib being final? Also additionally, I think we should have some existing logic for
when the 2 opposite ends where proximity triggers happen becomes too narrow to avoid multiple triggers. Now may also
be a good time to revisit some of that logic given this change." (4) "peek" — "it allows to act on a structure that
would be retroactively written where we would not have been able to act on. So it's a good thing."

1. Which sub is "the most recent counter sub" — **b: any live sub of the parent cycle whose direction is opposite the
   parent's, on either lens, the latest started.**
2. Which of its cycles — **a: only its most recent cycle whose BOS is confirmed by then.**
3. When a sub zone counts on an H1 candle — **a: only if known before the candle opens** (its M15 moment ≤ the last M15
   of the previous H1 candle).
4. When a sub POI counts — **leaning b: from its cycle's CTS_CONFIRMED (fib final)**; the user asked whether b keeps
   sub_3's POI (IC M15 2930) for the 820 trigger → YES: sub_3 cycle 2's CTS_CONFIRMED is M15 3086 (H1 771) → counts
   from H1 772 (rule 3); 749–771 the set is parent CTS + sub_3 c2 BOS 0.58171 (no hit: highest high 0.57968).
5. Fallback when the counter sub ends / reverses — **"Yes I think. … if the sub reverses, there would need to be a
   trigger on the other side before another trigger happens on the current side. Then at that point there will be
   another new counter that has started anyways. So I think it would be rare where the opp_sd zones will only be CTS
   zone"**. Checked: a bearish counter sub reverses only on a break of its BOS threshold (= its BOS zone OUTER, e.g.
   0.58196 for sub_3 c2), so price passes the BOS zone inner (in the opp_sd set) first → the opp_sd fires at or before
   the reversal candle. Edge cases where the fallback is reached: a narrow-mode cap blocks that opp_sd; the reversal
   lands in the same H1 hour its cycle's BOS confirms (the strict hour rule still shows the previous cycle).
6. "sub_4" across the reversal — **a: accept the zero-length (1,2) record.**
7. Architecture — **a: the H1 scan inside the lifecycle sweep.**
8. Scope — **the sd side stays parent BOS + parent POI only.**

## Findings — round 2 (2026-10-14)

- **When a fib is final:** at its cycle's CTS_CONFIRMED. `FibTracker.on_cts_confirmed` "lock[s] the Fib"; a confirmed
  CTS anchor never moves (CROSS_CYCLE_FIB_SPEC §3); a later `CTS_THRESHOLD_UPDATED` only drives the NEXT cycle's
  pre-established cross (`_on_cts_threshold_updated` targets cycle + 1). The IC pick reads candles only up to the fib's
  CTS anchor (+1: the pattern's confirmation, a gap's c3; fill horizon `check_to_idx = cts_idx`) — all before the
  CTS_CONFIRMED moment (PLAN_Q §3's argument). A cycle that never confirms its CTS → its POI never counts. Rule 4b =
  how the PARENT POI is already read: PLAN_Q counts it from the scan start = the parent cycle's CTS_CONFIRMED.
- **The narrow rules measure the wrong gap for the overlay.** Rules 1–3 (LANDMINES "Narrow-Cycle Rules 1+2+3 Are a
  Triple") test `|cts_threshold − bos_threshold|` (the structure EXTREMES, H1 50 pips); triggers fire at the INNER
  edges, and the dead band between the two sides = level gap − 2 × 8 pips. Parent-only level gaps are constant per
  cycle (inner edges never move) — (0,0) 222.2, (0,1) 69.2, (1,0) 30.4, (1,1) 59.4, (1,2) 50.4 pips (final threshold
  gaps 264.0 / 117.4 / 56.8 / 82.0 / 99.3). With the overlay (0,1)'s level gap falls 69.2 (683–748) → 45.9 (749–771,
  sub_3 c2 BOS) → 38.4 (772–833, sub_3 c2 POI) → 34.3 (after 833, "sub_4" c0 BOS); the "sub_4" POI would make it
  17.3 (a 1.3-pip dead band — the ping-pong shape) but rule 4b keeps it out until H1 935. The overlay makes the gap
  NON-monotone (a new sub cycle shrinks it, a replacement can widen it) — the triple's invariant "narrow → wide exactly
  once" does not hold for a level gap.
- **A latent bug the change exposes — the first_counter fallback window.** (1,2)'s `first_counter` (M15 3743) reads the
  confluence lens of (1,2), whose only record by then is the zero-length `first_confluence` → excluded
  (`_build_sibling_cts_ref_zone_from_pool`: `not r.is_zero_length`) → the fallback `_window_extreme_idx(m15_df, lo,
  hi, −lower_sd)` with `lo = 0` for first_counter (`_sibling_cts_idx_window`: "0 is a safe lower bound" — safe only
  for the pool read, NOT for the extreme on the shared frame) → the lowest low of the whole frame up to 3743 = H1 96
  (2025-11-20 22:00, 0.55808, sid_0's start). The fallback never fires today (0 "sibling-CTS unavailable" lines in the
  S_Q `run.log`). Bounded to the parent cycle (lo = the record's parent floor, M15 3611) the extreme = M15 3621 (H1
  905, 0.57112) = today's sub_5 input.
- **"sub_4"'s cross-fib limit changes source:** today `parent_cts_threshold` 0.57376 (started by (1,2)'s
  first_confluence), M = 0; started as a counter sub of (0,1) → `parent_bos_threshold` 0.5736 (like sub_3). Its cycle 0
  CTS anchor 0.57112 is beyond both → M stays 0 → no fib decision should move (to be confirmed by the build).
- **"sub_4"'s drawn window moves** from 3621–3819 (confluence lens) to 3335–3611 (counter lens), so its M15 chart
  zones change with it.
- **(1,2) under the agreed rules (1b / 2a / 3a / 4b), sd side unchanged (sd level = min(BOS inner 0.57996, POI bottom
  0.57786) = 0.57786):** up to 1035 the same triggers as today — 954 keeps its candle but its reference becomes sub_5's
  POI (IC M15 3654, top 0.57306; counts from H1 947 — sub_5 c0 CTS_CONFIRMED M15 3786 = H1 946), 4.5 pips (was
  opp_sd:CTS 6.9); 947–953 lows all above 0.57386. After 1035 the counter sub is sub_7: c0 (BOS 4031 = H1 1007, POI
  0.57484 from 4103) → c1 (BOS 4157 = H1 1039, inner 0.57455, POI 0.57462 from 4159) → **c2 (BOS 4179 = H1 1044, inner
  0.57728; no CTS_CONFIRMED → no POI) — 5.8 pips from the sd level, the two 8-pip bands overlap** → a PING-PONG:
  1045 opp_sd (low 7.0 pips above) → 1046 sd → 1048 opp_sd → 1049 sd (last H1 candle 1057) = 2 extra
  `subsequent_confluence` + 2 extra `subsequent_counter` subs. Today's narrow rules do not stop it ((1,2)'s threshold
  gap is 99.3 pips → wide).
- **Level gaps with the overlay:** (0,1) narrow from 749 (45.9) on any 50-pip level measure; (1,2) 50.4 (935–946) →
  48.0 (947–1044, sub_5 / sub_7 POIs) → 5.8 (1045+). The level gap ≤ the threshold gap always (every inner edge lies
  inside the structure's extremes; the closest-wins pick keeps the opp_sd level inside the CTS inner).
- **Narrow-guard options measured on this window** (H1, 50 pips where a threshold is needed):
  - keep Rules 1–3 as they are → 820 / 833 + the 1045–1049 ping-pong;
  - the level gap feeds Rules 2 + 3 → (1,2)'s CTS was confirmed by `sd_zone_proximity`, so Rule 2 blocks every
    narrow-mode trigger from 947 → 954 and 1035 vanish → sub_6 / sub_7 never built. REJECT;
  - the level gap feeds ONLY a Rule-3-style cap (at most one more sd + one more opp_sd per cycle while level-narrow;
    Rules 1 + 2 stay on the threshold gap) → (0,1) 820 / 833 kept (the first of each while narrow), (1,2) 954 / 1035
    kept, 1045–1049 removed;
  - a minimum-distance filter on sub zones (a sub zone joins only if its inner is ≥ N pips from the sd level) → N must
    be ≤ 38.4 to keep 820; N = 20 removes sub_7 c2 (5.8) → 1045–1049 removed, everything else kept.

## Round 3 (2026-10-14) — the user's answers + findings

The user (verbatim): "Fib Final Question — Ok on this we are good to use the final fib then" (4b DECIDED). "Q5. ok".
On the narrow rules: "Can you provide a description of what our current narrow rules are? … what happens when we judge
BOS to CTS is too narrow? … currently, I'm thinking of either something like a) requiring the 2nd sd side proximity
trigger and beyond to use a different reference instead. Could be the outer of the closest zone, or simply the inner
of the farther zone between BOS & POI. Could also be something like cascading increase on each additional trigger
b) Define some minimum boundary, and simply use that boundary as the reference for proximity triggers after the first
ones on each side when sd side zones overlap into this boundary. If they do not, then the sd side zones are still the
reference for proximity trigger c) When both BOS & POI zones exist and POI is closer zone, if after first sd trigger
via the POI zone, the price breaks past the POI zone then triggers opp_sd proximity, the 2nd sd proximity can no
longer use the POI zone, and must now use just the BOS zone. From all these options, currently I'm leaning toward c."
On Q6: "Does that not already exist? Do we not already have a rule that extends a sub across different parents? Why do
we need a new rule?" Decision 3 (two build steps, each with its own variant replay): "Sounds good".

- **The narrow rules never bind on H1 in this window:** the gap at each scan start (the CTS_CONFIRMED price vs the
  BOS_CONFIRMED price; it only grows after) is (0,0) 243.8, (0,1) 117.4, (1,0) 52.2, (1,1) 79.6, (1,2) 99.3 pips.
  Rule 1 (`market_structure._maybe_confirm_cts_via_proximity`) also carries the cycle-0 carve-out (`cts_cycle_id > 0`).
- **Option c measured** (the overlay as agreed, the sd side under c):
  - (0,1): 748 sd via the POI (low 0.57764); price then broke below the POI's outer edge 0.57676 (closes below from
    754; the deepest low 0.574 at 761) → 820 opp_sd → the next sd must use the BOS (inner 0.57494 → low ≤ 0.57574) →
    **872, not 833** (the first such low; the min low 821–901 is 0.57263 at 901). The subsequent_counter moves 833 →
    872 (its probe window becomes [LOH(820) = 3283, LOH(872) = 3491]).
  - (1,2): the POI holds 935–954 (no high above 0.5785) → 1035 sd via the POI unchanged. 1044 WICKS 0.6 pip above
    the POI's outer edge (high 0.57856, close 0.57825 inside); 1045 opp_sd (sub_7 c2 BOS). A WICK break → the next sd
    needs the BOS (high ≥ 0.57916) → **1049** → 1045 + 1049 (2 extra triggers, not 4; lows 1050–1057 stay above
    0.57808, so no further opp_sd on sub_7 — a counter sub started at 1049 is unknown). A CLOSE break → the first close
    above 0.5785 is 1047 → 1045 opp, 1046 sd (POI), 1048 opp, 1049 sd (BOS) = the same 4.
  - Without the overlay (S_Q itself) c moves nothing ((1,2)'s POI holds before 954; (0,1) has no second sd).
  - The (1,2) ping-pong is an opp_sd-side overlap: sub_7 c2's BOS zone (0.57691–0.57728) lies 5.8 pips under the
    parent POI, and 1045 is no rebound — price is CLIMBING into the parent's sd zones (closes 1044–1049: 0.57825 →
    0.57904) while its lows stay just above the sub zone.
  - Option a ("the farther zone" unconditionally for the 2nd+ sd) moves S_Q itself: (1,2)'s 1035 sd (POI) → 1049 (BOS)
    → today's sub_7 changes.
- **Q6 — the cross-parent continuation EXISTS (the same-candle handover):** a sub stays live across parent cycles
  when the next parent cycle's record starts on (or before) the candle its previous record ends (`_phase1_record_start`:
  strict `<`, "== t is the same-candle handover"). S_Q example: sub_2 — its (0,0) reversal record ends 2611
  (`parent_end`), its (0,1) first_confluence record starts 2611 → one continuous sub. "sub_4" misses it by a GAP: its
  (0,1) counter record ends 3611 (`parent_end`); the (1,2) first_confluence record starts at 3621 = max(probe finalize
  3621, trigger 3611, parent floor 3611) → the sub ended 10 M15 candles before → frozen zero-length. What would be
  new is bridging that gap, not crossing parents.

## Open — round 2

1. The narrow guard (options + their effect on this window: "Narrow-guard options" above): keep Rules 1–3 / a
   Rule-3-style cap on the level gap (recommended; everywhere or only when a sub zone sets the opp_sd level) / a
   minimum-distance filter on sub zones / both; its pip value; the commit granularity (with the opp_sd change or its
   own step).
2. The first_counter fallback window: bound it to the parent cycle (a separate, zero-delta fix landed first) and/or let
   the sibling read see a zero-length record's sub.
3. H1-only runs: the opp_sd set = the parent CTS alone (assumed acceptable).

## Round 4 (2026-10-19) — the user's direction

The user, verbatim: "Thank you for checking on the narrow rules today. Those look good in terms of what they are
doing currently. I have a better sense now of what I want to implement, and it will be a combination of option c)
and a new set of narrow rules. I will paste the detailed write-up below. The write-up will also address a few other
items." The write-up: `user_writeup_round4_20261019.md` (this folder, verbatim).

**Decided in the write-up:** option c generalised to a per-side REFERENCE LADDER (a zone is passed by an H1 close
beyond its outer; the side moves to the next-closest inner; the parent BOS / CTS is the last rung and locks); same-sd
(confluence) sub zones join the sd side only when the parent has a POI and they lie beyond it; a second narrow rule on
the REFERENCE GAP (|sd ref inner − opp_sd ref inner|, rounded, `<=` 40 H1 / 25 M15 / 12 M5) applying Rule 3 only
(1 sd + 1 opp_sd), with a per-side counter reset when that side's update widens the gap; Rules 1–3 stay. Q6: keep
today's pool (no bridge; the "sub_4" retrigger is a zero-length record).

### Findings — round 4 (measured on S_Q; simulator `r4_sim.py`, ILLUSTRATIVE — S_Q's subs, not the new world's)

- **Proximity pips are 8 / 5 / 3 in code** (`zone_proximity.py:91-95`), not 9 / 6 / 3; MS reads the same value for
  the sd-proximity CTS confirmation (`market_structure.py:87-88`). 9 pips gives the same trigger candles in the
  simulation of (0,1) / (1,2), but changing it is its own cause.
- **Zone inners never move:** 0 of 39 KL zones (H1 10, M15 counter 8, M15 confluence 21) change their inner; outers
  expand (H1: 3 CTS zones; M15 confluence: 4 BOS zones). The pass test must read the outer as of that candle.
- **The user's (0,1) numbers check:** sub_3 c1 BOS 0.58366 from H1 708 (record start M15 2829), c1 POI 0.58279 from
  712, c2 BOS 0.58171 from 749 (gap 45.9 → 46), the 754 close 0.57657 < POI outer 0.57676 → sd ref = BOS 0.57494 from
  755 (67.7 → 68), c2 POI 0.58096 from 772 (60.2 → 60), c3 BOS 0.58055 from 898 (56.1). Chain: 748 sd:POI,
  **820 opp_sd:sub_3 c2 POI, 872 sd:BOS** (the 2nd sd is 872, not 830–833: the POI was passed at 754). Never narrow.
- **(1,2) with S_Q's subs:** 935 sd:POI, 954 opp:sub_5 POI (gap 48), 1035 sd:POI; 1036 sub_7 c0 POI 0.57484 → gap 30
  NARROW; 1040 sub_7 c1 → ref 0.57462, gap 32 (a new counter-sub cycle WIDENED the gap — rule 2a drops c0's closer
  POI); 1045 c2 BOS 0.57728 → gap 6; 1045 opp, 1046 sd, 1047 close 0.57852 > POI outer 0.5785 → sd ref = BOS 0.57996
  (gap 27, sd counter reset), 1048 opp BLOCKED (opp counter 1) → the 1048 / 1049 ping-pong gone. If the counter
  counted the cycle's earlier triggers (954, 1035) instead of only those since narrowing, 1045 / 1046 are blocked too.
- **Lock must be "reached by passing", not "is the ref":** lock-by-default kills the overlay — (0,1) only 748 (the CTS
  is the ref 683–707, before sub_3 joins (0,1)); (1,2) 954 stays on the CTS (936–946 the CTS 0.57282 is closer than
  sub_5's BOS 0.57271).
- **Parent zones only, every main cycle:** exactly S_Q's 6 triggers ((0,0) none, (0,1) 748, (1,0) none — parent gap
  30.4 narrow but nothing reaches, (1,1) 820 / 871, (1,2) 935 / 954 / 1035) → the ladder + reference-gap rule on
  parent zones is zero-delta on S_Q (a clean first build step).
- **Existing Rule 3** counts only triggers fired while narrow (`zone_proximity.py:515-519`), never reset.
- **(1,2) same-sd zones (S_Q):** sub_4 POI 0.57885 qualifies (beyond the parent POI 0.57786, inside the BOS 0.57996);
  sub_4 BOS 0.58055 lies beyond the parent BOS (never reachable — the BOS locks first); sub_6's zones (0.57758 /
  0.57653) do not qualify. In the new world sub_4's (1,2) record is zero-length (Q6) → not live under (1,2).
- **"Would have been" time already exists for events:** `ev.idx` = the structure's own moment, lifecycle ignored; the
  scan runs on it (why the collapsed (1,1) fires 820 / 871). Zone records carry the CLAMPED time: sub_3 c0 BOS event
  2649 vs zone `confirmed_idx` 2829; sub_4 c0 BOS 3306 vs 3621; sid_1 (1,1) CTS_CONFIRMED 810 vs zone 902; POIs have
  `poi_unclamped_activation_floor_idx`. Missing: subs of collapsed parent cycles are never built (no records under
  (1,0) / (1,1)) → their overlay would be empty; on S_Q no sub would qualify there anyway.
- ~~Q6 = keep ⇒ the first_counter fallback bug fires in the new world~~ — **CORRECTED 2026-10-19 (PLAN_R shadow):**
  the (1,2) first_confluence re-find is an APPROX probe-cache hit (finalize 3491) → its record starts at 3611 = the
  sub's (0,1) parent_end → the same-candle handover; "sub_4" stays live, the sibling read finds it, the fallback is
  NOT hit (0 `sibling-CTS unavailable`). The fix stays a latent-bug fix (PLAN_R §2).

### Decisions — round 5 (2026-10-19; the user's answers, verbatim excerpts)

1. Pips: "If in the code we are using 8/5/3, then it means some documentation are outdated. If that is the case, then
   I would reconsider the thresholds for gap that we would use to below: 1H: 40 15M: 25 5M: 15. We should also check
   anywhere in documentation where we have different values for this parameter and update those." → the
   reference-gap thresholds are **40 / 25 / 15**; proximity stays 8 / 5 / 3.
2. Lock only by passing: "Yes you are correct on that."
3. + 4. The counter: "Count would start AFTER cycle becomes too narrow from not narrow previously. So any triggers
   fired before becoming too narrow does not count. After becoming too narrow from not narrow, each additional update
   that expands the gap will reset the counter on the side where the update happened if we are still too narrow after
   the update. If the additional update is further reduction in gap once we are already too narrow, then we do not
   reset the counter. As soon as we become not too narrow, the counter is no longer needed. … Not narrow → reduce to
   narrow and counter resets → expansion resets counter on update side if still too narrow & reduction does nothing →
   expansion to not narrow & counter is gone → reduce to narrow and counter resets → …" (cause-agnostic: any
   widening update resets that side).
5. The pass defaults: "Yes". **Reopened 2026-10-20 (plan review):** the last default ("closes count from the scan
   start, even for a sub zone that only becomes usable later") contradicts the same reply's tables (simulated with
   "from when usable") and kills the overlay → PLAN_R §7 Q1.
6. Same-sd subs: "This should be based on unique subs, so essentially unique subs with the same direction as the
   parent. 2 & 3 are yes" (most recent cycle only; known before the H1 candle opens). On the sub POI's start: "I think
   POI should be able to exist before CTS_confirmed. … Fibs effectively won't change unless CTS updates before
   confirm, at which point Fib would simply update. … So I'm not sure what you mean by its POI only from its cycle's
   CTS_CONFIRMED" — OPEN (round-1 Q4 had chosen "from CTS_CONFIRMED"; explained in round 5).
7. POI beyond BOS: "Yes. … when we get to parent BOS as reference, it cannot be further expanded upon because passing
   this zone would effectively mean being close to reversal or actually hitting reversal."
8. Both narrow rules side by side: "Ok".
9. "Would have been" time — name the clock, park collapsed-cycle subs: "sounds good".
Q6 / 0 / 10: asked to explain the fallback-window fix again; the order decision waits on it.

### Findings — round 5

- **Stale 9-pip values fixed (uncommitted):** LANDMINES "Concrete cases" note, 11 comments in
  `tests/test_zone_proximity.py` (all still true at 8 pips; 31 / 31 pass), 2 memory notes (May history: 9 / 6 / 3
  from 2026-05-11, 8 / 5 / 3 since `804d19d` 2026-05-28). Canonical value: GLOSSARY `proximity_pips` (correct).
  CODE still stale: `market_structure.py:91` `_FALLBACK_PROXIMITY_PIPS = 9` ("matches H1 default") — reached only by
  callers that build `MarketStructure` directly (9 test files); every production build goes through
  `structure_engine._make_market_structure`, which always passes the TF value. Unregistered quirk: the probe's Phase 2
  MS runs at the H1 value (8) on M15 (`structure_engine.py` docstring "a deferred cleanup").
- **Sub POIs exist before CTS_CONFIRMED and move:** between a sub POI's first activation and its cycle's CTS_CONFIRMED
  the CTS updates often (sub_3 c2: active 3027, 6 CTS_UPDATEDs 3042–3047, CTS_CONFIRMED 3086; sub_4: active 3621
  (its start), CTS_UPDATED 3621, CTS_CONFIRMED 3741; sub_6: active 3819, CTS_UPDATEDs 3852–4000, CTS_CONFIRMED 4157).
  The saved POI is the FINAL pick; the in-flight pick is not stored. On S_Q, using the POI from its first activation
  changes no trigger: (0,1) 757–771 max high 0.57736 < 0.58016; (1,2) 936–946 min low 0.57641 > 0.57386.
- **The fallback path:** `_sibling_cts_idx_window` gives first_counter `lo = 0` (safe for the pool query, which is
  scoped to the parent cycle) but the fallback `_window_extreme_idx` scans the RAW M15 frame → the lowest low since
  the data start. The fallback builds a BOS-style ad-hoc zone (`build_ad_hoc_bos0_reference_zone`), the sibling path
  a CTS-style one (`_derive_cts_zone_ad_hoc`) — same input candle 3621 after the bound, the zone may differ: measure
  sub_5's start before landing.

### Decisions — round 6 (2026-10-19; verbatim)

1. The sub POI as a reference: "Ok a is correct" → usable from its cycle's CTS_CONFIRMED (final), as agreed in round 1.
2. Collapsed cycles: "OK we can go with parent zones only for now and consider subs later" → a register line
   (would-have-been subs for collapsed parent cycles), detail here.
3. The build order 0a / 0b / 1 / 2: "Sounds good".
4. The probe's H1-pips-on-M15 register line: "What is this? And why do we need it?" — explained in round 6: inert by
   design (`unified_probe.py:549-552`: the probe reads two CTS_ESTABLISHED, cycle 0's CTS_CONFIRMED and a reversal —
   never a proximity confirmation; cycle 0 cannot be proximity-confirmed); the only reason is bookkeeping.

### Raised in round 6 (open)

- **Crossed references:** an opp_sd sub zone can form on the far side of the current sd reference. Near miss in S_Q:
  (1,2) 1045 — sub_7 c2 BOS inner 0.57728 only 5.8 pips under the sd reference 0.57786 (1044 wicked to 0.57856,
  above it). Proposed: a signed gap (crossed ≤ 0 → narrow) and an opp_sd sub zone counts only while its inner lies
  strictly on the opp_sd side of the current sd reference.
- **The chart:** a trigger on a sub zone has no line on the H1 chart; the hover must name the source (parent / sub_N,
  cycle, BOS / POI / CTS, inner); optionally a stepped "current reference" line per side — needs a rendered example.
- **Rename scope:** internal trigger meta keys `prior_cts_prox_idx` / `this_cts_prox_idx` / `cts_prox_idx` /
  `cts_prox_zone_kind`, local `cts_prox_t` (multitf/ only, not saved) + PART4 §4.3.4 / §4.3.5 wording + the hover's
  `opp_sd:CTS`.
- **H1-only runs** (`lower_timeframes` without M15, `orchestrator.py:639`): parent zones only.
- **Ties** on equal inners (sub_7 c1 and c2 POIs share IC 4131 → 0.57462): only the label / pass state differs.
- **/compare:** the new subs renumber every later sub (`number_subs_by_start`) → compare by (direction,
  starting_idx), not sub_id.

### Decisions — round 7 (2026-10-19; verbatim)

- The probe register line: "Ok sounds good. We can add it" → added (register §E).
- Crossed references: "a) is correct" → a SIGNED reference gap (crossed ≤ 0 → narrow); a crossed opp_sd sub zone stays
  a candidate (the "exclude while crossed" filter, option b, NOT adopted).
- The chart: "Sounds good" → step 2's hover names the source zone; a stepped current-reference line per side rendered
  as a sample at the chart look, not locked before.
- "Rest sounds good": the rename scope (internal `*cts_prox*` names + PART4 wording in step 1), parent zones only on
  runs without the M15 pass, ties (parent zone first, then sub zones, BOS before POI within a sub), /compare keyed by
  (direction, starting_idx).
- Register §G line added for the collapsed-cycle subs (round 6, decision 2); the `_INBOX.md` line drained.

### Raised in round 7

- **The change TRIGGERS register §E "Opposite-direction POIs"** and falsifies POI_ZONES_SPEC "POIs are always
  sd-direction (invariant)" → "Consequence" ("opp_sd zone proximity triggers only consider the CTS KL zone … no POIs
  participate"). The per-structure invariant still holds (a POI is in ITS structure's direction); the parent's opp_sd
  side now reads a counter sub's POI. PLAN_R updates the spec paragraph and closes the register item (sides assigned
  explicitly by the sub's direction relative to the parent; the main's own `poi_zones` stay sd-only).
- Names for the new pieces; the main's own sd-proximity CTS confirmation stays on parent zones (circular otherwise);
  a newest counter sub without a usable zone yet (S_Q: 8 / 8 subs had BOS_0 confirmed before their record start).

### Decisions — round 8 (2026-10-19; verbatim)

1. Naming: "since we have used the term reference in other places. I think we should add the word "proximity" to all
   the naming so we know specifically everything refers to proximity triggers" (the clash is real: the probe's
   `ReferenceZone`, the IC `ic_ref_price` / `ic_ref_source`). Names: sd / opp_sd **proximity reference**, **proximity
   reference ladder**, **passed proximity zone**, **proximity reference gap** (signed), **narrow proximity-reference
   cap** (`DEFAULT_MIN_PROXIMITY_REFERENCE_GAP_PIPS` = 40 / 25 / 15), **proximity overlay** (the sub zones added to a
   side), **opp_sd zones proximity trigger**; code keeps the existing `prox` abbreviation (`prior_opp_sd_prox_idx`).
   Unchanged: "narrow-cycle Rules 1–3"; the clocks (anchor / moment / lifecycle) are not proximity-specific.
2. The POI-spec paragraph + register §E "Opposite-direction POIs": "Sounds good".
3. The main's sd-proximity CTS confirmation stays on parent zones: "Yes and this is why we don't consider sub zones
   that are "above" the parent POI. And if no POI, then we don't consider sub zones at all and only use BOS. This would
   also ensure that CTS confirmation does not consider these subs whether POI exists or not."
4. ~~The newest counter sub without a usable zone → the newest live counter sub with one, else the parent CTS: "Yup
   sounds good".~~ **SUPERSEDED round 9** by the hold (A) — the shadow showed this default restarts the ping-pong at
   H1 1047 → 1048 / 1049 (below, "Plan drafting").

### Plan drafting (2026-10-19) — PLAN_R written: `engine_v2/plans/PLAN_R_opp_sd_zones_proximity.md`

- Shadow runs (`r5_shadow.py`: today's pipeline, the (0,1) / (1,2) proximity lists replaced by the round-4 chains):
  subs 8 → 11, records 11 → 15, unresolved 4 = 4; "sub_4" = (−1, 3304) re-created by 872's subsequent_counter,
  live 3491 → 3819 across the handover (the user's round-1 expectation held); full table PLAN_R §5.4.
- **Correction to round 8 #4's reassurance:** "8 / 8 subs had BOS_0 before their record start" holds for S_Q only —
  the new run's sub from 1046 (start 4183, record 4187) confirms its BOS at 4189. With the round-8 default the opp_sd
  side falls back to the parent CTS at H1 1047 (gap 50, wide → counters dropped), then narrows at 1048 with fresh
  counters → 1048 / 1049 fire again (the ping-pong returns). OPEN: PLAN_R §5.3 — (A) hold the previous sub's zones
  until the newcomer has a usable zone (recommended), (B) a narrow episode ends only through a pass, (C) accept.
- Step 0b measured: the 9 direct-MS test files pass 242 / 242 with `_FALLBACK_PROXIMITY_PIPS = 8`.

### Decisions — round 9 (2026-10-19; verbatim: "A hold, keep 0a, go ahead with the review")

1. PLAN_R §5.3 = **(A) the hold**. Rendered precisely (PLAN_R §1 "Which sub feeds a side"): each side keeps a STICKY
   proximity source sub; it moves forward only to a newer sub (this side's direction, this parent cycle, record
   started) once that sub has a usable zone. So a sub's zones leave the ladder only by being passed or superseded by a
   newer sub's usable zone, never merely because the sub ended or was replaced. (1,2) prediction: 1045, 1046, then
   nothing through 1057 (1048 blocked by the opp_sd counter; 1049 by alternation).
2. Step 0a (the sibling-CTS fallback window bounded at the parent cycle's M15 floor): **kept** — a zero-change
   latent-bug fix.
3. The plan review: **go** (stated beforehand: about 3 read-only agents, 0.5–1M tokens).

### Plan review (2026-10-20) — 3 read-only agents, 0.84M tokens; folded into PLAN_R in place ("plan review")

- **Measurements:** step 1 zero-change re-verified twice, independently (6 triggers; passes 754 / 1047; gaps 50.4 /
  5.8 / 17.8). Corrections: the first_counter fallback pick is 130.4 pips low (not "about 146"); 7 test files reach the
  MS fallback (not 9; the full suite passes 1743 / 1743 at 8); step 2 DOES reach the sibling-CTS fallback once — 1046's
  subsequent_counter (its sibling, 1045's sub, has 0 events) → sub (1, 4183)'s start 4183 — so my "the fallback is
  never reached" (2026-10-19 reply) was wrong for that path.
- **Rules fixes adopted (faithful to the user's words):** the last rung is never pass-tested; outward resets judged per
  side (round 5); the strict most-recent cycle (round 1 Q2); the sd source switch needs a qualifying zone; an ended
  source sub is frozen at its end; gates block only the trigger test; narrow-at-scan-start = a narrowing; the outer as
  of the candle's close.
- **Sweep design fixes:** the tick at (`LOH(h−1)`, after SUB_END) — H1 1057 has no M15 bar; zones read as of their
  own usable moment (the natural-end view leaks post-end cycles; the POI pick is not mechanism-guaranteed final);
  usability from the events; every pre-sweep reader of the trigger lists switched (the sub WVMI attribution, the df
  attrs, the count log, the early return); the hover in both charts, branching on the source first.
- **Open → the user:** Q1 when a sub zone's closes start counting for a pass (from usable — recommended — vs from the
  scan start, which kills the overlay); Q2 the opp_sd lock (stay on the CTS vs re-open for a newer counter sub).

### Decisions — round 10 (2026-10-20; verbatim)

1. Q1 (when a sub zone's closes start counting for a pass): "ok let's go with a)" → from the candle the zone becomes
   usable (a parent zone: the scan start). Supersedes the round-5 default (#5, last bullet).
2. Q2 (the opp_sd lock): "yes b) is correct" → a NEWER counter sub's usable zone re-opens the opp_sd side after a lock
   on the parent CTS; the sd lock on the parent BOS stays permanent.
3. Asked: "When the sd side switches to a newer sub: only once that sub has a zone beyond the parent POI. Can you
   provide more details on what this means?" — answered with (1,2)'s data (PLAN_R §5.3: "sub_4" = (−1,3304) c0 POI
   0.57885 qualifies; old sub_6 = (−1,3760), zones 0.57758 / 0.57653 inside the parent POI, would otherwise displace it
   at H1 955; same triggers either way, the sd reference differs 1048–1050).
4. "I think it would probably be better to start the build in a new session right?" → yes; handoff prepared
   (the simulators copied into `sims/`, this folder).

### Decisions — round 11 (2026-10-20; verbatim)

1. Q3 (what re-opens the opp_sd side after a lock on the parent CTS): "yes let's go with ii)" → a newer counter sub's
   usable zone OR the current source sub's newer cycle's usable zone. The sd lock on the parent BOS stays permanent.
2. Q4 (commits + saves): "ok" → 4 commits (0a; 0b + the uncommitted docs / plan / Q&A / `sims/`; 1; 2), each after its
   own `/compare` (0a / 0b / 1 = 0 / 27 vs S_Q); `/commit-save` only after step 2.
3. "Before updating memory & documentation and handoff prompt, anything else worth bringing up or discussing? Just
   doing one final pass" — answered 2026-10-20 (the final-pass list in the reply).

### Decisions — round 12 (2026-10-20; verbatim: "combined review is fine, go ahead with memory and handoff")

1. Landing reviews: ONE combined review of 0a + 0b + 1 (2 agents, ~0.4–0.6M) + the step-2 review (3 agents,
   ~0.8–1.2M); total ~1.2–1.8M, stated beforehand (PLAN_R §6).
2. Final-pass notes given to the user: step 2's FIRING chain is simulated (the pool outcomes are measured) — a build
   that differs stops and explains, never absorbs; the stepped "current proximity reference" line is a chart-look
   SAMPLE for the user to judge.
3. The build starts in a NEW session (the kickoff prompt written 2026-10-20). PLAN_R status: READY TO BUILD.

### The build session — round 13 (2026-10-20; the landing review of steps 0a + 0b + 1)

- **Found by the code reviewer (latent, not on this window):** step 0a's `lo = max(lo, the real-time floor)` makes the
  sibling window EMPTY (`lo > hi`) for a trigger fired before its parent cycle's floor — possible only when a
  retroactively established cycle still has a real-time lifecycle (e.g. had (1,2) been established at H1 910, (1,1) —
  established 748, CTS confirmed 810, floor 902 = M15 3611 — would be live 902 → 910, and its 820 first_counter /
  871 subsequent_confluence would get `[3611, 3283]` / `[3611, 3487]` → skipped). On S_Q every live cycle's floor ==
  its CTS-established moment ((0,0) 115, (0,1) 652, (1,2) 902); only the collapsed (1,0) / (1,1) (703 / 748) differ.
- The user asked: "Can you provide more details on this? What exactly is the floor you are referring to? Also what does
  it mean that it can invert the sibling window? And how can a trigger fire before the parent cycle's real-time floor
  (again what is the meaning of real-time floor here?)" — explained (the floor = the cycle's real-time lifecycle start,
  max(its establishment, its structure's real-time start / reversal handoff); the window `[lo, hi]`; the scan runs on
  the structure's own clock, so a retroactive cycle's triggers precede its floor).
- **Decision: "Establishment moment (Recommended)"** — the fallback window's `lo` is bounded by the parent cycle's
  CTS-established MOMENT (the moment clock; M15 LOH of `ParentTables.cts_moment`), not the real-time floor: never empty
  (every trigger comes after its cycle's CTS_CONFIRMED ≥ that moment), identical on this window. Example: 820 →
  `[2995, 3283]`, 871 → `[3283, 3487]`. Built as the step-0a follow-up (PLAN_R §2, §8).
- Also from the review: the stale 9 / 6 / 3 table in `MARKET_STRUCTURE_SPEC.md` (step 0b's doc sweep missed it); the
  "CTS-prox class" / "opp_sd-prox class" double name; 5 test gaps on step-2 paths (opp_sd / overlay passes, an
  opp_sd outward reset while narrow, an sd = −1 cap case, the outer read at the pass site, the gap-None path) + "Rule 2
  is not part of the cap" pinned only incidentally — fixed in the review follow-up commit.
