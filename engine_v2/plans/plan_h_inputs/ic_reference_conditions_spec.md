# The IC reference conditions — the user's write-up (verbatim, 2026-10-09)

Item 1's first rule experiment (PLAN_H §8.9): two new IC candidate requirements, applied alongside the §3.1 base
conditions BEFORE the institutional-pattern check (the user, 2026-10-09: "These new requirements may change the
candidates that make it to the IP check"). They replace POI_ZONES_SPEC §3.2's per-scenario tables (written before the
cross-cycle fib revamp — "we will do it a bit differently given all the changes we've made since I original wrote
those"). This file is the user's text, unedited; the engineering reading and every decision live in PLAN_H.

---

New Requirements / Conditions.
The IC must be completely below (when sd is +1) the previous structure BOS price or previous cycle CTS price that the current cycle / trend broke out of, and the IC must be before the anchor candle of the pattern that break the previous struct BOS or previous cycle CTS. There are 2 different scenarios it would apply.

1. Whenever a structure is in +1 direction and forms cycles 1+, the newly established CTS_x will need to break past the CTS_x-1 price. Now assume here we only have single Fibs on cycles 1+. The reference price / line for cycle x Fib & IC would be the CTS_x-1. The cycle 1 IC candidate must be entirely below the CTS_x-1 level and the IC candidate must be before the anchor candle of the pattern that breaks CTS_x-1. One important note here is that the price we use is the original CTS anchor price that was confirmed. We do not use cts_threshold, so even if the CTS price updates, we use the original confirmed anchor price. This is different than when we used cts_threshold as reference price for determining where cross-cycle fibs can extend to, and it is intentional.

2. While the 1st scenario is the general case whenever we form a new cycle, the 2nd scenario deals specifically with right after a reversal, or when the price breaks the prev structure bos_threshold. After the reversal, whenever a new cycle forms, both the previous cycle's CTS and the previous structure's most recent BOS can be potential reference prices. The reference price we choose will be the one that's further toward the opposite direction of the current MS (post reversal). After a reversal, we may also have cross cycle fibs anywhere from cycle 0 to the first cycle that breaks past the previous bos_threshold. Depending on the size of the cross fib, the reference prices will also differ.

The 1H chart can help demonstrate this, and it is also an example in the -1 direction. The previous BOS here would be 591 and we have break of the bos_threshold to form reversal around 902. Here cross-cycle fibs are allowed from cycle 0 to cycle 2 since CTS_2 is the first that crosses the bos_threshold.

Just for hypothetical purposes, let's assume there is a valid cross-cycle fib from 0-2. Here when the cross-fib takes up the entire structure from cycle 0-2, we cannot / do not use any prev cycle CTS as reference price. So in this case prev BOS is the only reference price, and IC candidate must be both above the prev BOS price or 0.5736 entirely AND the IC candidate must be before the anchor of the valid reversal breakout pattern. In the event that cycle 0 Fib is actually single, then there will also be no prev cycle CTS price to reference.
If we have a valid cross-cycle fib from 1-2, then we can use CTS_0 as reference price. Now we have both prev struct BOS and prev cycle CTS available as references. When both reference prices are available, we always use the one that's closer to the opposite direction of the structure. In this case, it would be the CTS_0 at 710 (0.57902), which is higher than 0.5736. The IC must be above that price, and before the candle pattern that breaks CTS_0 and forms a new CTS_established.
Finally, if we just have a single fib on cycle 2, then the reference prices are CTS_1 and prev struct BOS. Here the closer opposite direction price is CTS_1 (idx 761) or 0.57400. The IC candidate must be above this price and before the pattern anchor that breaks CTS_1.

Once the reversal structure has formed a 2nd CTS past the previous struct's bos_threshold, then we no longer have any more cross-cycle fibs, and we no longer use prev structure BOS as a reference price. Thus from this cycle and forward, the rules will be just what we have from Scenario 1. Again, the reference price we use should be the original confirmed CTS anchor & BOS anchor prices. We do not use bos_thresholds or cts_thresholds which could change from the original anchor values (different from cross-cycle fib limit reference price)

So to recap, after a reversal, from cycle 0 to the max cycle x where a cross-cycle fib is allowed, prev struct BOS will always be one of the reference prices. Once cycle x+1 is established as the 2nd CTS that's moved past the prev bos_threshold and we can no longer have cross-cycle fibs, prev struct BOS can no longer be a reference price. At cycle 0 and/or if cross-cycle fibs extend to cycle 0, then we won't have a prev cycle CTS reference price until a valid Fib no longer starts from cycle 0, but otherwise prev cycle CTS will always be a reference price as long as there is a previous cycle. Whenever we have both reference prices available, we always use the one that's further in the opposite MS direction.

Now any details you'd like to discuss or clarify? Any questions you have? Any nuances that I may have missed? Any issues that could potentially arise with implementation?
