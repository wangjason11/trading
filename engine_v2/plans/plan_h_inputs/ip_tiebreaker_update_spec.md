# The institutional-pattern tiebreakers update — the user's write-up (verbatim, 2026-10-09)

> **Superseded in part by PLAN_P (2026-10-13):** the first tiebreaker below (the immediate unfilled imbalance)
> became IC condition 3 — `../PLAN_P_ic_condition3_step1a.md`, POI_ZONES_SPEC §3.1 / §3.4. The text below stays verbatim.

Item 1 (one POI per cycle), before the user's holistic IC write-up: a NEW first tiebreaker among institutional
patterns tied on the same rank and shape; today's TB1 (the IC's candle type) and TB2 (the most recent IC before the
earliest unfilled imbalance after the oldest tied IC) move down to TB2 / TB3. This file is the user's text, unedited;
the engineering reading, the measurements and every decision live in PLAN_H (§6, §8.9).

---

Thanks. Ok before my full writeup and any of the other tasks, first I want to make an update to our Institutional Pattern (IP) tiebreakers when we end up with patterns of the same tier & ranking. We will add an additional tiebreaker and move the other current two down.

Updated Tiebreakers
1. Prefer IC immediately followed by unfilled imbalance, “immediately” here means within 2 candles after the IC candidate (<= IC_idx + 2). This is essentially a filtering condition (true/false). All true IC candidates move onto tiebreaker 2 to be further narrowed down. If no trues, then all candidates move onto 2. If only 1 true, then that’s the final IC & and we skip the rest.
2. Prefer maru > normal > pinbar (Current TB1)
3. Prefer the most recent IC prior to the oldest (earliest) unfilled imbalance that comes after the oldest IC candidate (Current TB2)

To provide some theory, I want to add this as the 1st tiebreaker because it in essence is a stronger & more stringent version of one of the 4 conditions we use to identify IC candidates prior to identifying IP. That condition only requires there to be unfilled imbalance anywhere after the IC candidate. This tiebreaker is much more stringent, requiring the unfilled imbalance to be within 2 candles. But ultimately, this condition is a really strong signal because it indicates immediate and large movement away from the IC, which indicates strong momentum in the move, which then also theoretically reaffirms the IC is a strong “Point Of Interest” (POI) that may provide resistance if/when price retraces after the initial move. But ultimately, this by itself is not as strong of a signal as the institutional pattern, which is why it is applied as a tiebreaker after IP. However, in some ways this also has similarities to how we define IP. When the continuation leg is formed via stacked candles, those will often naturally form imbalances which would automatically satisfy this condition, and from a strength perspective, those patterns are relatively weaker. Now what this new tiebreaker does help with would be in scenarios where we have multiple same ranked patterns & IC candidates along with certain imbalances getting filled that were initially unfilled. See the below sequence:

imbalance1, IC1, IC2, imbalance2, imbalance3, IC3, imbalance4

Assume all these IC candidates are all part of the same ranked IP and have the same candle type classification. Assume all imbalances currently are unfilled. When we get to current tiebreakers, TB1 does not further narrow down the candidates. Using TB2, the final IC is IC2: the oldest IC candidate is IC1, the oldest unfilled imbalance after is imbalance2, and the most recent IC prior is IC2.

Now hypothetically let’s say imbalance2 was actually filled instead of being unfilled. TB1 still does nothing. TB2’s flow: oldest IC is IC1, oldest unfilled imbalance after is imbalance3, and most recent IC prior to imbalance3 is IC2. Same result, nothing changed. Finally, let’s assume that IC2 is in close proximity to imbalance2, that IC3 is in close proximity to imbalance4, and lastly that imbalance3 is far from imbalance2. IC2 would be the current tiebreaker winner, but its own immediate trailing imbalance was filled. There was another unfilled imbalance afterwards, but that imbalance is much farther away and cannot provide support or reaffirmation of an IC that was so far in the past. Tiebreaker 1 helps solve this by prioritizing ICs where the immediate subsequent imbalance is still unfilled.

Therefore, in the above example when imbalance2 gets filled, the new TB1 would filter it out and ultimately leave IC3 as our final IC, so we always prioritize ICs that come immediately before. With TB1, even if imbalance2 remains unfilled it would also already filter out IC1, since that IC has no immediate imbalance. That would then also impact the now new TB3:  oldest IC is now IC2 instead because TB1 has now eliminated IC1, oldest unfilled imbalance after is imbalance2, and most recent IC prior to imbalance2 is IC2. When imbalance2 gets filled, TB1 would filter out both IC1 and IC2. What you are left with then is IC3 now being the oldest and only IC candidate.

I don’t think this should be too complicated of a change, and with it, I think we would wrap up the Institutional Pattern process. Afterwards, I will get into the remaining full picture. But now do you have any questions around the new update? Anything you’d like to clarify or discuss? Any feedback or potential issues / conflicts you foresee?
