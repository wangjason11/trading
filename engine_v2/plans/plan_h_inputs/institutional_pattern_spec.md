# Institutional-pattern IC — the user's spec (Week 9 Part 1 item 1)

> **Repo copy, canonical from 2026-10-01** (the original, `artifacts/institutional_pattern_prompt.md`, is gitignored).
> The user's text, AMENDED IN PLACE as the discussion decides: each amendment carries its decision ID in [brackets] →
> `../PLAN_H_institutional_pattern_poi.md` §6 (decisions) — §1 there holds the precise engineering reading.
> **Naming [N1/N3]:** initial wave → **initial leg**; pullback wave → **opp leg**; continuation wave → **continuation
> leg**; "anchor candle" = the initial leg (its first candle = `pattern_anchor_idx`, GLOSSARY "Naming Standard");
> "end candle" = the continuation leg (its last candle = `pattern_end_idx`); the rank label = `ip_rank` (e.g. "2A.i").
> The prose below keeps the original words.

In our current logic to find IC candles & build POI zone, we can potentially have multiple ICs on a wave (BOS to CTS) based on % of overlap with the 61.8 - 80 zone. However, this muddies the POI zone signal. I would like to update the original logic to now generate a single IC candle and single POI zone per BOS to CTS cycle. 

As for general POI zone build and institutional candle (IC) identification logic, I want to emphasize that all other logic and conditions remain the same, including that the IC must be inside the 61.8 - 80 Fib zone. The only part we are changing is using the most recent candle meeting each % overlap threshold to determine the IC [N4: the original said "earliest" — a typo]. The logic will now be that as long as any part of an IC candidate overlaps/touches in any capacity the 61.8 - 80 Fib zone, it qualifies as a candidate. So the 61.8 - 80 Fib zone condition becomes just another filter / gate for remaining IC candidates after all other previous rules are satisfied (unfilled imbalance, IC candle direction etc.). This change by itself would not reduce ICs, but instead increase the number of candidates. Therefore, we will introduce a new logic to identify the single IC to create a single POI zone per cycle.

Some final house keeping notes before diving into details:
1. This updated logic should be applicable to any TF or main/sub structures
2. We should as much as possible stick to established naming standards for patterns / structures. If at certain points I name things that deviate from our conventions, you should alert me. Also, certain concepts like “pullback wave” may be mentioned and used here. We need to ensure we keep the definition of that clean and separated from “pullback” elsewhere. For concepts like breakout and pullback in general, they are easy way to convey movements in various different places, but we need to ensure each place we are using the correct technical definitions. 
3. In general if anything is ambiguous, unclear, or misinterpreted, please be sure to make a note and have us discuss. 

The Institutional Pattern
There is an ideal “institutional pattern” we are looking for based on 3 or more candles that form a “mini N pattern” (when sd=+1) with 3 segments: initial wave confluence with sd, pullback wave counter to sd, and then continuation wave confluence with sd. The institution candle (IC) will be the last (opposite sd) candle in the pullback wave of this pattern immediately before the first candle of the continuation wave. The ideal pattern we are looking for has this IC sandwiched between 2 movements of the same direction as the BOS to CTS wave (or opposite direction of the IC), thus forming this “min N” pattern. 

There are different variations of this pattern, and depending on the characteristics of the pattern, there are also strong and weak cases. Additionally, in the long term there may be some discretionary aspects to identifying the IC that might not be perfectly conveyed in adhering to stringent rulesets. For now, let me explain in general what we are looking for.  

A. Base Variants

The most basic ideas around this pattern are captured here:
[`institutional_pattern_strong.png`](institutional_pattern_strong.png) (this folder)

Don’t get caught up on the specific categorization of the strong / weak cases in these examples just yet. I will discuss those in detail (actual categorizations will be slightly different than the ones in the file). For now, these figures will give you a general sense of the pattern we are looking for. These are the simplest and easiest to identify variants of the institutional pattern.

The attachment demonstrates bullish patterns in a bullish MS, but the same concept applies for bearish patterns & MS. The base variants have 3 candles, where you have 2 maru candles in the same direction as the MS that anchor (initial wave) and end (continuation wave) the pattern. The IC will be located in between these maru candles, and will be the last opposite direction candle (pullback) before the pattern end maru candle. The strongest versions of the institutional pattern are exactly 3 candles, with 1 candle for each segment. This is the institutional pattern we are looking for, where the IC is effectively the last downward pressure before a significant continuation move in the MS direction. Finally, for a true institutional pattern, it is also important that the end candle of the pattern is also big_maru with a bigger body than the middle / pullback wave candle body.. 

So thus far, the pattern requirements are:

1. Anchor (initial wave) & end (continuation wave) candles confluence with MS direction
2. Anchor candle / initial wave must be maru
3. Middle candle / pullback wave between the anchor & end candles must be in opposite direction (will be the IC)
4. End candle / continuation wave must be big_maru. Additionally, the big maru candle’s body must be bigger than the body of the pullback / middle candle (also the IC) 
(5). The IC identified as the last opposite direction candle prior to end candle must overlap with 61.8 - 80 Fib Zone

Requirement 5 technically is specifically on the IC candle as opposed to just this pattern, so I think that condition / requirement can live separately outside the institutional pattern logic (since it is a requirement even outside the institutional pattern) [D4, 2026-10-02: the principle for ALL IC conditions — they stay outside the pattern and are applied first; the pattern search runs last around the surviving candidates. The never-implemented POI_ZONES_SPEC §3.2 scenario IC filters stay in that spec, unapplied, for future testing.]

Now for the priority rankings of different variations within this pattern, below outline additional conditions the pattern must meet. Furthermore, unless the conditions directly differ from the above mentioned 5 requirements, we always assume those still have to be met and that the conditions below are additive. 

1A. big_maru-maru-big_maru (strongest)
1. Anchor candle also big_maru
2. Middle candle is maru

1B. maru-maru-big_maru (strong)
1. Middle candle is maru

2A. maru-normal-big_maru (medium)
1. Middle candle is normal

3A. maru-pinbar-big_maru (weak)
1. Middle candle is pinbar (or essentially not normal OR maru)

So here even with the basic pattern, we have a tiered system, where tier 1 are the strongest patterns, down to tier 3 which are weaker. Within each tier, there also exist subgroups (ABC) ranked from strongest to weakest variants of the pattern. So when we search these patterns, we always prioritize the higher ranked based on tier & subgroups and only consider weaker patterns when stronger patterns cannot be found. 

These essentially cover all the variants of the simple base version of the pattern that are all shown in the 1st attachment I mentioned. Later on as we add additional variants of the institutional pattern, we will add them into various spots in the established tier system. Within each tier, new variants will also be ranked accordingly relative to the other variants in the same tier. 

We can discuss one of those additional variants right now. Normally, the institutional pattern requires the anchor to be maru. However, we can make one exception (only for the anchor candle) and allow it to be a big_normal if it’s not maru. This is something we can accept for the base institutional pattern even though it breaks a requirement. As a result they are also ranked weaker.

2B. big_normal-maru-big_maru (medium)
2C. big_normal-normal-big_maru (medium)
3B. big_normal-pinbar-big_maru (weak)

[N2] "big_normal" here = candle type **normal** AND the big_normal length ratio (a big pinbar does not qualify); likewise the end candle = candle type **maru** AND the big_maru length ratio.

The patterns we’ve discussed so far are the simplest variants of this institutional pattern, where each segment of the pattern (initial wave, pullback wave, continuation wave) is represented by 1 candle. But the same concept would apply to patterns where each segment spanned across multiple candles. When we are unable to identify the simple base variant of the pattern or in some cases when not all the requirements for the simple patterns can be met, we look for multi-candle variants of the institutional pattern.

B. Multi-candle Variants

For multi-candle variants, we are essentially going to reproduce the same initial wave, pullback wave, and continuation wave pattern. However, now instead of each segment having 1 candle, each segment can be represented by multiple candles that are all in the same direction per segment.

A version of this idea is captured in this figure here:
[`institutional_pattern_weak.png`](institutional_pattern_weak.png) (this folder)

However, note that this figure does not fully capture all the possibilities of these variants, but simply demonstrates the specific case where the middle segment now consists of 2 candles instead of 1, both in the same direction.

Recall our previously set 5 requirements / rules for the institutional pattern when the segments were single candles. Below would be the modified rules for the multi-candle pattern

1. Initial wave & continuation wave candles all confluence with MS direction. Both waves can have a max of up to 3 consecutive same direction candles.
   - [P4] A leg is always the MAX consecutive run available, up to 3 — never a shorter pick: the initial leg = the most recent (up to) 3 sd candles immediately before the opp leg; the continuation leg = the first (up to) 3 sd candles immediately after the IC. E.g. 5 consecutive sd candles before the pullback wave → the most recent 3; exactly 3 → those 3; 1 sd candle preceded by a −sd candle → 1.
   - [B1] the initial leg may not start before the fib's BOS anchor; [B2] the continuation leg may not run past the fib's CTS anchor; [B1a] a run crossing either is cut there.
   - [V1, added 2026-10-02] Not only candle direction: every leg candle's VOLUME direction (`vol_dir` = close vs the previous close) must agree too — initial & continuation legs: candle AND volume direction with sd; opp leg: both counter to sd. This accounts for candle gaps that make the volume direction differ from the candle direction; it applies to single and multi legs, and a candle failing it breaks the run (so some consecutive same-direction candles are broken).
   - [V1a, 2026-10-02] Volume direction 0 and candle direction 0 are treated consistently: when one of them is 0, the other decides — volume direction 0 → the candle's direction; a doji (candle direction 0) → the volume direction. Only a candle with BOTH at 0 is a doji for the doji rules below (a "true doji"). Names: PLAN_H N5.
   - [VD, 2026-10-05] A doji WITH a volume direction (a "volume-only doji") is an ordinary directional candle: it belongs to the leg of its direction, never breaks it, and can be the IC (as the last opp candle). The doji rules below apply only to true dojis, which are truly directionless.
   - [P6 / P6a–P6e / P6c-final / J1] True dojis do not break a leg, and are full members of the leg they are ASSIGNED to: they count toward the max of 3 (IC, doji, up, up, up → the continuation leg is doji-up-up) and make that leg multi. ~~a pattern may start on a doji~~ [E1, 2026-10-05: a true doji never starts or ends the pattern — it is never the first candle of the initial leg nor the last candle of the continuation leg, even if those legs are already multi and their caps have not been met; such a doji is left out (up up doji up up before the opp leg → the initial leg is up-up).] What matters is assigning each doji to the right leg — a wrong assignment can erroneously make a single leg multi. A doji between the opp and continuation legs always joins the continuation (the IC is never a true doji, so a doji is never the last candle of the opp leg: up up up down doji up up → the opp leg is the single down, the continuation doji-up-up is multi). A doji between the initial and opp legs [J1, 2026-10-05] is tried in EACH leg and the better pattern (rank, then shape) wins, each placement following that leg's cap rules: joining an initial leg already at the max pushes its earliest candle out (the max-3 window slides), and where a pattern allows only a single initial leg (3D) the doji cannot join it. ~~(was: the initial leg if it has not hit the max consecutive candles, else the opp leg)~~ — e.g. maru · doji · maru(IC) · big_maru: doji → initial gives maru-maru-big_maru (multi-single-single), doji → opp gives maru-multi-big_maru (single-multi-single, weaker) → the doji joins the initial leg. [E1] A doji at the pattern's outer edge is never part of it (above).
2. Initial wave candles when “stacked” together must be maru (stacking explained below)
3. Pullback wave candles are between the initial wave & continuation and must all be in the opposite direction of MS [P5: the opp leg = the whole run of consecutive opposite candles ending at the IC]. There is no cap on the number of candles for pullback wave. The last candle of the pullback wave prior to the first candle of the continuation wave will be the IC candidate.
4. Continuation wave when “stacked” together must be big_maru. Additionally, the “stacked” body length must be bigger than the body length of the “stacked” pullback wave (stacking explained below). 
(5). The IC identified as the last opposite direction candle of the pullback wave prior to the first continuation wave candle must overlap with 61.8 - 80 Fib Zone.

Stacking
I will now explain “stacking”. Because we require each segment to all be the same direction candles, we can effectively line these candles up together at the same point on the x-axis to form a larger “stacked” single candle, end over end, for each segment. The stacked candle’s OHLC (open, high, low, close) would be defined by the open of the first candle in the stack, the close of the last candle in the stack, the high of the entire stack of candles, and low of the entire stack of candles. If we stack all candles from each segment, visually it would once again become the same base variant pattern of 3 single candles, and that’s how we would then apply the same candle requirements for requirements 2 and 4. [P2 / item 5, 2026-10-02] A stacked candle is defined exactly like a single candle — the same classifier (maru / normal / pinbar incl. the special-maru promotion and the per-TF pip floor) and the same big_maru / big_normal parameters.

Initial wave: the stacked candle body length vs. candle length ratio must meet maru threshold. In our exception case for big_normal, it then must meet the big_normal definition/ratio as a stacked candle. 

Continuation wave: the stacked candle body length vs. candle length ratio must meet maru threshold. The total stacked length must meet the big_maru threshold. Our existing rule uses the previous 5 marus to establish the threshold. ~~The previous 5 will be prior to the first candle in the wave segment, so essentially the standard set for big_maru would be as of that candle.~~ [P3, amended] The previous 5 marus are counted before the first candle of the WHOLE pattern (the initial leg's first candle) — the same as the other candle patterns — and the same reference serves the initial leg's big_maru / big_normal. Finally, the body length of the continuation wave must also be greater than the body length of the pullback wave.

Because these multi-candle variants are effectively also the same as the base variants once the candles are stacked, they can also be placed into the different tiers we discussed accordingly. However, their specific placements within the tiers will be more nuanced, while other cases will have exceptions to the general rule. 

Some general rules of thumb
1. In general, a single candle segment is always stronger / more preferable than a multi-candle segment
2. For multi-candle segments, the number of candles does not really matter (e.g 2 candles is not stronger than 3 candles)
3. If middle segment is more than 1 candle, it instantly becomes weak and forms its own subgroup
4. Continuation wave is more important, so between 2 patterns where they both have just 1 multi-candle segment, one on the initial wave and one on the continuation wave, the one with multi-candle continuation wave is weaker. 

Below are the updated tiers & rankings now ordered based on these rules.

1A. big_maru-maru-big_maru (strongest)
i. single-single-single
ii. multi-single-single
iii. single-single-multi
iv. multi-single-multi

1B. maru-maru-big_maru (strong)
i. single-single-single
ii. multi-single-single
iii. single-single-multi
iv. multi-single-multi

2A. maru-normal-big_maru (medium)
i. single-single-single
ii. multi-single-single
iii. single-single-multi
iv. multi-single-multi

2B. big_normal-maru-big_maru (medium)
i. single-single-single
ii. multi-single-single
iii. single-single-multi
iv. multi-single-multi

2C. big_normal-normal-big_maru (medium)
i. single-single-single
ii. multi-single-single
iii. single-single-multi
iv. multi-single-multi

3A. maru-pinbar-big_maru (weak)
i. single-single-single
ii. multi-single-single
iii. single-single-multi
iv. multi-single-multi

3B. big_normal-pinbar-big_maru (weak)
i. single-single-single
ii. multi-single-single
iii. single-single-multi
iv. multi-single-multi

3C. maru-multi-big_maru (weak) ← this is a new established tier when the pullback wave is multi-candles. 
i. single-multi-single
ii. multi-multi-single
iii. single-multi-multi
iv. multi-multi-multi

3D. big_normal-multi-big_maru (weak) [P8, added 2026-10-01] ← a new established subgroup when the pullback wave is multi-candles.
i. single-multi-single
ii. single-multi-multi
For this one, we only allow a single candle on the initial leg because that is already an exception; allowing multi would be even weaker, so we exclude them. [P8a / J1 / D3c, 2026-10-05] Its initial-leg cap is effectively 1 (multi is not allowed): for 3D the initial leg must be just the single big_normal candle before the opp leg — never a "stacked" big_normal; that candle may still be part of a multi (stacked) initial leg for a higher-ranked, non-3D pattern, and the better pattern wins. A doji after it cannot join a 3D initial leg (it has already met the cap); it joins the initial leg only if that (stacked with earlier candles) yields a pattern better than 3D, else it joins the opp leg.
[P8] Any combination not listed above (e.g. a plain normal / pinbar initial leg) is not a pattern.

Note again, there is no ranking difference based on how many candles are in a multi candle segment within the same tier & subgroups. For example, if 2 maru-multi-big_maru patterns have the following candle segments: 2-3-2 and 3-3-3. Both of these are multi-multi-multi, and there are no inherent ranking difference between these 2. However, there is a difference between 1-1-3 and 2-1-2 (the first is single-single-multi while 2nd is multi-single-multi), where 1-1-3 would be ranked higher. 

If we have multiple patterns found of the same most granular ranking, the last step is using additional tiebreakers to filter to a single IC and POI (used in order): 

~~1. IC: prefer maru > normal > pinbar~~
~~2. Prefer candle closer to BOS …~~
~~3. Prefer candles with larger total length~~
~~4. Prefer candles with larger body length~~

~~[TB, 2026-10-02 round 2] 1. IC: prefer maru > normal > pinbar 2. Prefer candles with larger total length (if within 80% then tie; a param) 3. Prefer candles with larger body length (if within 80% then tie) 4. Prefer more recent candle~~

[TB2, updated 2026-10-02 round 3 — replaces both lists above, incl. R4's "closer to BOS" and the 80% bands]
1. IC: prefer maru > normal > pinbar
2. Prefer the most recent IC prior to the oldest (earliest) unfilled imbalance — [TB2a, 2026-10-05] the oldest unfilled imbalance AFTER the oldest tied IC is the reference, and the IC is the most recent tied IC prior to it. E.g. tied ICs 4083 and 4136 with unfilled imbalances 4029, 4132, 4151, …: the reference is 4132 → 4083; had 4029 and 4132 not existed, the reference would be 4151 → 4136. An IC may be the candle that opens the gap (also a multi-candle gap) — possible only when that candle is opposite to sd, since ICs are. "Unfilled" = the same test as the IC's own unfilled-imbalance condition.

These tiebreakers always reach 1 IC (2 never ties). ~~In most cases, we won’t need to go further than 2. [R6] An exact tie after all 4 → the earlier candle.~~

In the scenario that after we’ve exhausted all variants and cannot find an institutional pattern, for now that would simply indicate the cycle does not have a POI zone [X1, 2026-10-05: the old V30 / V60 / V90 selection stays in the code, unused, as a possible fallback for no-IP cycles — tbd]. However, in future iterations we will build an additional process where we would go into a lower TF to find the same institutional patterns that we cannot find on the current TF. This is partly why when we build multi-candle initial & continuation waves, we cap the number of consecutive candles for now to 3. With the current TFs we have (1H and 15M), since 1H consists of four 15M candles, the higher TF should already be able to capture any institutional patterns that would have formed via 4 candles in the 15M TF since we are stacking. Down the line, when we jump from 15M down to 5M TF, for the 5M TF, we will likely cap initial & continuation wave consecutive candles to 2. For now, for both 15M and 1H tf, 3 candles will be the cap for multi-candle segments. 

Finally, here are some of my thoughts in terms how to incorporate this into already existing logic. As previously mentioned, we still need all the other requirements to be true to find an IC and form POI zone. Now every -sd candle within the 61.8 - 80 Fib zone is a potential candidate [VD, 2026-10-05: −sd = the agreed direction (V1a), so a volume-only doji with volume direction −sd is a candidate and a candle whose volume direction opposes its candle direction is not]. From there, we can then try to find the institutional patterns around these candidate candles. I’m not sure if that’s the most efficient and least costly way, but I’m obviously looking to your guidance here as the engineering expert to implement this. 
