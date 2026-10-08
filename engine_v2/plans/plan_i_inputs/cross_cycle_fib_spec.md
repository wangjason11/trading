# The cross-cycle fib revamp — the user's spec (Week 9 Part 1 item 2)

> **Repo copy, canonical from 2026-10-07** — the user's text, pasted in the revamp session (session 1 of 3: the rules
> → the spec; PLAN_H §6 LR3). Everything below the line is VERBATIM: part 1 = the rule, part 2 = the user's answers to
> the 8 inputs listed in the session (main vs sub differences). Amendments are made IN PLACE only as the discussion
> decides, each carrying its decision ID in [brackets] → `../PLAN_I_cross_cycle_fib_revamp.md` §6 (decisions); the
> precise engineering reading lives there (§1), not here.
> **Amended 2026-10-07 (round 1):** the bracketed [Q1]–[Q8] notes below = the decisions of PLAN_I §6; the
> user's words are untouched. **Round 2 (2026-10-07):** the user's feedback is part 3 (verbatim, at the end); its
> previous-sub rule supersedes [Q1] (bracket below marked). **Rounds 3–5 (2026-10-07):** parts 4–6
> (verbatim, at the end). **Round 6 (2026-10-07, session 2 — the questions the code audit raised):** part 7. **Round 7 (the four
> follow-ups; the plan approved):** part 8.

---

The main change I want to make an update to is limit the farthest cycle a cross-cycle fib can go. This will apply to both main and sub. In general the update logic will be: cross-cycle fibs can form starting from cycle 0 up to the first cycle whose CTS_anchor exceeds (>) a reference price / threshold in the structure’s direction. [Q2: strictly — reaching the price is not exceeding; the first such cycle is the last cross target, `M`] On the 2nd cycle whose CTS_anchor exceeds the reference price, that cycle can no longer be cross-fib and from then on all further cycles are single. Another way to think about it is that if a particular cycle of a parent is formed by the 2nd+ CTS established (price) beyond the reference price (meaning there was already 1 CTS established price that exceeded the reference price), all those cycles will be single fibs. 

For the CTS_anchor terminology here, we would accept both the confirmed CTS anchor and also the most recent CTS_updated (this would be more reserved for scenarios on the sub when say CTS is never confirmed, but somehow the cycle later ends). For consistency, can you check how we currently use CTS_anchor? Is it explicitly the CTS anchor idx of a confirmed CTS? Or do we also reference it in instances that apply to the most recent / max CTS_updated.

Specific Example:
If cycle 0’s CTS_anchor is past the reference price, then it is just a single cycle, and future cycles are all singles. 
If cycle 0’s CTS_anchor sits below the reference price and cycle 1’s CTS anchor is above, then we would allow 0-1 cross-cycle fibs. Cycle 2 and onward would be single. 
If 0 and 1 anchors both sit below reference price and cycle 2’s anchor is beyond, then allow 0-2 cross-fibs, and cycle 3+ are single. 

So we are simply putting a limit to where the cross-cycle fibs can go up to, afterwards any additional cycles will have single fibs. For places where we allow cross-cycle fibs, all other logic in terms of when & conditions that they are active, etc. should all still remain the same. This includes parts around unfilled imbalances, and how we shrink cross-fibs or default to single fibs on newer cycles if imbalances are filled (e.g. 0-1 cross-fib could be come 0-2 cross-fib, or become 1-2 cross-fib, or simply single fib on cycle 2). 

Now as for what that reference price is, that will different depending on the trigger of the MS. We will go through each trigger below:

Main
When we run MS very first time on main to start trading: no reference price. For this trigger started MS, we will not have cross cycle fibs. [Q6: its cycle 0 gets a single fib — every structure's cycle 0 is single] 
Reversal: the most recent bos_threshold that a valid pattern broke out of that triggered the reversal. The reversal logic should be a unified change that applies to both main and sub the same. For main, it should be relatively straightforward.

Sub
For subs, to be consistent with our other implementation, everything around fibs & cross-fibs should be on the unique sub level. However, given our updated pooled structure, the entire unique sub’s structure we will accept based on its first trigger. Any retriggers assumes already existing sub structure created by the first trigger. So mirroring that, the unique sub’s reference threshold price for cross fibs will be set depending on the very first trigger (in actual realtime) mapped / assigned to the unique sub. [R6-2: computed when the sub's MS runs — its first trigger's fire, from that trigger at its record's start — and re-checked against the sub's actual start after the sweep, a WARNING on any mismatch]

Sub Reversal
So same as main, if unique sub’s first trigger is reversal, we use the most recent bos_threshold that a valid pattern broke out of that triggered the sub reversal. 

Sub First confluence
If mapped parent cycle id is 0, there will be no reference price, and we will also not have cross cycle fibs on this unique sub.
Otherwise, reference price will be most recent CTS_threshold that a valid pattern broke from to result in CTS established, which is the trigger for first confluence. Since this covers all parent_cycle_id > 0, there should always be a prior CTS_threshold we broke out of. [Q3: the parent's cycle c−1 `cts_threshold` as of parent cycle c's CTS_ESTABLISHED]

All other sub triggers (first counter, subsequent confluence, subsequent counter)
The reference price we use will be the most recent BOS_threshold of the previous unique sub. [Q1: the most recent previous unique sub of the OPPOSITE direction; its `bos_threshold` at the trigger candle; none → no reference, no cross — SUPERSEDED by R2-1 (part 3): the previous sub x−1 at x's start_idx; same direction → its confirmed `cts_threshold`, opposite → its `bos_threshold`] 

And that is the updated cross-cycle fib rule I want to implement. Now I’m not sure if I’m remembering correctly, but we may have discussed something similar to this previously or maybe even have written a previous iteration somewhere (I’m not entirely sure). At any rate, it’s very likely even if we did, the above full logic differs from what I previously came up with. 


The 8 inputs look good and are good callouts on things to address. I will do so now where it’s relevant

1. This will still be slightly different between main & sub, but it will be more unified with the above logic I think. Rather than applying logic based on sid. We now attach variations to triggers. Going forward, we should move away from using the Scenario X terminology since again, those don’t provide enough information to understand them in the names. But I’m not sure what is Scenario 1 here. At any rate in the updated logic, on main the start trading trigger cannot have cross-cycle fibs. While on subs, first confluence on a parent cycle 0 (on any parent sid) cannot have cross cycle fibs. Outside of sid 0, all other sids will be triggered by main reversal. Thus a parent sid 1+, parent cycle id 0 will always be the first cycle after a reversal. However, in order for the first confluence of parent cycle 0 to trigger, the reversal needs to have happened first. Thus the unique sub for any parent cycle id 0 first confluence trigger will not be that unique sub’s first trigger, and we can almost effectively ignore using any cross-cycle fib logic. 
2. Early cross: I think this we can actually unify and have on main as well. Main’s only trigger that allows cross-cycle fib is reversal, and given our updated logic, early cross should be allowed within the confines that a cross-cycle fib would be allowed after reversal. 
3. [Q4: REMOVED — Scenario 1 and the revert; the limit replaces them] We could potentially remove? Or merge somehow? In theory, the new logic with reference / threshold price is what will cap how far the cross-fibs can go.
4. [Q5: the sub's current-candle fill check everywhere] Hmm not sure on this one just yet. Any thoughts after reading the rule change I proposed?
5. Here I would also want to bring what we have for sub to main. Specifically, how we deal with fibs after unique sub first triggered via reversal should be same as how we deal with fibs post reversal on main.
6. [Q8: main adopts the sub's fib meta] What does bespoke mean?
7. [Q7: MS's in-flight POI read uses the same fib decision as the drawn fib] This one I’ll need more details around. I’m not quite understanding fully 
8. We shouldn’t have coexistence with the updated logic

After looking at these inputs you brought up, I’m inclined on 4 and 7, where we may have differences between sub and main to unify based on what we have in sub. Even without our current edits, the sub logic is already more advanced. Now that we can tie the only main trigger that allows for cross-fib (reversal) to the same sub trigger, it should help with merging?

Now given the above explanation, do you have any questions? Anything you want to clarify? Any nuances that I may have missed that should be addressed? Any potential conflicts or issues with other existing processes that may cause difficulties down the line? Any places where the wording I’m using may cause future confusions? For concepts, some times I may use / to denote that we can use either term for naming purposes, but I also look to you to help suggest to make naming consistent with our established principles. Any thoughts on names as well?

---

## Part 3 — round 2: the user's feedback on the decided rules (2026-10-07, VERBATIM)

> Decision IDs: `../PLAN_I_cross_cycle_fib_revamp.md` §6 R2-0 … R2-5, R2-N, R2-Q3. Its point 3's second paragraph
> (the previous-sub rule) SUPERSEDES part 1's "The reference price we use will be the most recent BOS_threshold of the
> previous unique sub" and its [Q1] note.

Feedback / comments on the decided rules
1. Sounds good all cycle 0 will have single fib
2. Sounds good
3. On first_confluence on parent cycle c ≥ 1, do we have some similar concept as applied breakout’s cts_frozen? If so that would mirror reversal. Just curious on this.
On first_counter / subsequent_confluence / subsequent_counter: let me adjust the rule here. I actually want do something that’s different.

In live realtime, at the start_idx (moment) of unique sub (sub_id=x),
- If sub_id=x-1 most recent / last / ending direction is the same as sub_id=x, then reference price will be most recent / ending cts_threshold attached to the most recent confirmed CTS anchor (do not use CTS anchors that are only established or updated without having been confirmed)
- If sub_id=x-1 most recent / last / ending direction is opposite of sub_id=x, then reference price will be most recent / ending bos_threshold attached to the most recent BOS anchor

This logic makes more sense. The reference price should always be coming from the x-1 unique sub, but depending on what that sub’s direction is compared to x-1, we would either use bos_threshold or cts_threshold.

4. Sounds good. I would still like to have the logic documented somewhere for historical safekeeping and so we can reference the iterations we’ve gone through in the future if needed.
5. Ok
6. Ok

Still open
1. Can you provide definition for the cross_limit variables you proposed? And yes retire Scenario 1/2/3, but still keep a record for the future. Everything else looks ok
2. This edge case shouldn’t happen. In the updated logic, if previous sub is opposite direction with no bos_threshold, then we simply use the previous sub’s finalized starting_idx as fallback.
3. Sounds good. In terms of anything else:
- this should already be implemented, but fib activeness should be clamped by the cycle’s lifecycle so that doesn’t change
- the reference price is determined once at the unique sub’s start_idx moment, and then it does not change


Let’s hold off for end of the session.

Anything else worth clarifying or any other edge cases or potential issues you see worth discussing?

---

## Part 4 — round 3: the user's detailed responses (2026-10-07, VERBATIM)

> Decision IDs: `../PLAN_I_cross_cycle_fib_revamp.md` §6 R3-1 … R3-6 (R3-2 and R3-4 open at the time of writing; answered in part 5).

Let me give you a more detailed response to each of the points you madee

On cts_frozen, ok gotcha. Thanks for the explanation. So here I just want to double check would the cts_threshold, also kind of “freeze” in the same manner? Requirements for CTS established (when there is a previous CTS and thus CTS_threshold), are not that different from reversal. Essentially, there needs to be a pattern that breaks / closes beyond that cts_threshold. Therefore, there are also scenarios where the final confirmed breakout pattern can apply several candles later. Thus there are different instances where cts_threshold would update (failed breakout) similar to how bos_threshold would expand, vs. when cts_threshold would be frozen to wait to see if there would be breakout. Even if cts_frozen as a variable does not exist, the concept / process should be similar / mirror each other for both reversal and breakout.

X-1 rule change.
Ok a few things I want to clarify. You had listed:

counter sub 5, buy
conf sub 6, sell
sub 7, buy

For 5 & 6, you put counter and conf in front, which confused me a bit. Are unique subs not independent of lens? So having counter sub 5 doesn’t mean there’s also a conf sub 5 correct? You are simply indicating the the lens where we chart the sub? There should be only 1 unique sub 5, 1 unique 6, and 1 unique 7 correct?

Also what about sub 4? Why did you not list that one out? What would that unique sub be started by?

Cross limit definitions.
I see. Let’s use cross_fib_limit_xxx, so that it is more intuitive? Cross by itself sounds a bit confusing
I’m still not quite sure what the source values are and what this field is supposed to show?

R2-a — what "x−1" means.
Ok, so this goes back to whether a unique sub can exist in time without having a start_idx. You are saying for sub 0, the trigger happens at 463, but start at 1020. So the question goes back to in realtime moments from 463 - 1019, do we have a triggerRecord existing with start_idx NULL, and along with that do we also have a unique sub recorded with start_idx NULL? Again trigger dt is meant to log where a historical even would have happened and not the same as lifecycle start moment.

Also just because x-1 start_idx comes after the trigger, does not mean by the time x starts (again not starting based on trigger but based on lifecycle), that x-1 would not have already started. I feel like it should be the case if we are considering lifecycle, at each new unique sub x’s start_idx, sub x-1 should have started, which is why I feel i) should work. But I’m open to confirming & double checking.

R2-b
X-1’s latest as fallback is ok.

---

## Part 5 — round 4: the user's replies to round 3 (2026-10-07, VERBATIM)

> Decision IDs: `../PLAN_I_cross_cycle_fib_revamp.md` §6 R4-1 … R4-7. (The quoted "Only first_confluence subs can lag"
> was my round-3 wording — corrected in PLAN_I §3.4: the parent floor is a second lag source.)

1. Thank you on verifying. We are good here
2. I see. Thank you for confirming on sub 4.
3. Got it. Sounds good thanks
4. I see. Thanks for your explanation. So it is possible to have a unique sub with start_idx NULL but not possible for a record to exist with start_idx NULL.

“Only first_confluence subs can lag; every other type starts at its trigger”. On this, does this imply that we would only have this issue of x-1 not started at the start of x, if x-1 was a first confluence? If that’s the case again, I think that’s ok because by definition what followed first confluence would be first counter, and by the time first counter & the sub starts, it should always be the case that the first confluence sub has started.

On R3-2. yes I agree with this change. Unique sub numbering should be based on life cycle start.
- A sub that never starts (all its records zero-length) would get no id; today it holds one. On this point, can you clarify? So you are saying currently a unique sub that never starts would get a unique sub id? That doesn’t sound right? Or are you saying that these exists in a particular moment in time, but that all numbered unique subs would eventually have a start_idx? But if it’s the case there are unique subs numbered that never starts, then that shouldn’t be happening even now.

Still Open
R3-2. Ok
R3-4. keep

Nothing off the top of my head currently on my side, but things might come up down the line. Are there anything else that you feel like we should address?

---

## Part 6 — round 5: the user's replies to round 4 (2026-10-07, VERBATIM)

> Decision IDs: `../PLAN_I_cross_cycle_fib_revamp.md` §6 R5-1 … R5-3. (The first line quotes my round-4 reply.)

Even when x−1 is a first_confluence sub, the first_counter doesn't wait for it.
On this point here, yes there is no explicit wait for first_confluence to start by first counter. But by trigger & probe end logic, first counter subs can’t start before first confluence subs. The triggering condition as well as the probe end_idx for first counter is zone proximity to sd zones, meaning the the probe end_idx would be the start_idx. For first confluence the probe end_idx is CTS confirmed, which means CTS confirmed is the latest that the first confluence would start (other conditions may result it starting earlier). CTS confirmed by definition is either a zone proximity to sd OR a valid pullback. And so logically, if CTS confirmed was via zone proximity, then both first confluence & first counter would have the same start_idx, meaning when x starts, x-1 also has started. If CTS confirmed is a valid pullback, then it must be the case that the price has not pulled back near zone proximity yet. Therefore, in this case the pullback also must have happened before proximity, meaning also that x-1 must have started before x. This is all based all innately laid out in the logic so no explicit enforcement is necessary. That’s my take.

Ok I agree with the R3-2 fix, but in general I’m still of the belief that without it the numbering should still hold because of the some of the logic I laid out. However, I think your fix also doesn’t change things and is more correct so let’s go with that.

R4-5. Yes this is important to keep

R4-6. Yes, when I said “ending” it is for situations where the x-1 structure has ended, and at that point we would assume those bos or cts thresholds at that point stops updating, and new thresholds would be tied to new BOS or CTS in newly started structure x. Keep in mind that when the structure ended due to say new structure (reversal) or new parent (new parent cycle), the ending thresholds we should be using are the frozen ones we discussed.

But yes both defaults are fine.

Does everything else look good? Any last items we should talk about?

---

## Part 7 — round 6: the user's answers to the session-2 questions (2026-10-07, VERBATIM)

> Decision IDs: `../PLAN_I_cross_cycle_fib_revamp.md` §6 R6-1 … R6-7 (the questions: §5 "Round 6"). Every answer takes
> the recommendation offered.

R6-1. ok sounds good
R6-2. ok
R6-3. your recs sounds good
R6-4. your rec sounds good
R6-5. yes
R6-6. your rec sounds good
R6-7. ok

Anything else you need or want to bring up?

---

## Part 8 — round 7: the user's answers to the four follow-ups (2026-10-07, VERBATIM)

> Decision IDs: `../PLAN_I_cross_cycle_fib_revamp.md` §6 R7-1 … R7-4 (the questions: §5 "Round 7").

1. yes
2. sounds good on both
3. neutral sounds good
4. sounds good.

Anything else you want to bring up on your end? Are we ready to start the build? Is ultracode needed? If everything's all set we can commit
