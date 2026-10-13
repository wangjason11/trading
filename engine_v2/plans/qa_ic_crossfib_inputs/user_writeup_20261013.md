# The user's write-up (2026-10-13, session 2) — three IC-identification changes, VERBATIM

Pasted by the user after the Q1–Q3 findings (`qa_findings.md`), for review and discussion before any plan. Kept
verbatim (the rationale is the canonical record of WHY); decisions taken on it are recorded in the owning plan.

---

Here’s our current Step 1 filter to identify IC candidates:
1. Candle must be in opposite direction of MS
2. Candle must be entirely below (when sd is +1) the BOS or CTS reference price, and it must be prior to the anchor candle of the pattern that breaks the reference BOS or CTS
3. There must be unfilled imbalance after the candle
4. The candle must overlap with the 61.8 - 80 Fib zone

I've come to realize that requirement 3 is actually a very loose condition that is very similar to the Fib level requirement that a cycle must have general unfilled imbalance. Step 1 requirement 3 is a bit more specific since it has to reference a specific candle as the IC candidate, but these 2 rules in different places somewhat serve the same purpose and are thus a bit redundant.

Instead of the current requirement 3 under Step 1, what I want to do instead is take one of our tiebreak / general ranking rules and replace the current requirement 3. Specifically the rule: Prefer IC immediately followed by unfilled imbalance, within IC + 2 (<=)

Now previously this immediate imbalance rule was a “preference” rule meant to rank multiple different candidates. Now when it is moved under Step 1, it becomes a “must have” requirement that filters out any candidates that do not meet it. The intended effect is to make Step 1 more stringent and further narrow down what candidates can proceed to the following steps. From reviewing the charts, the previous requirement 3 allowed certain candles that had unfilled imbalance far after the IC candidate to pass onto Step 2, where we were able to find institutional patterns and eventually become the final IC. However, upon review, these are actually not preferable vs. IC candidates that would have immediate imbalance after but no IP. Having immediate imbalance is actually a stronger indication of a good IC and where future price likely would retrace to. Thus, before even applying IP search, I want to keep the candidate list to ONLY candles where we have immediate imbalance after rather than just any generic imbalance anytime after the IC. Again, having imbalance in general in a cycle is already required via Fib levels, and isn’t as important of an indicator of IC strength as having IMMEDIATE imbalance after the IC candidate.

I believe this change should move the IC for sub_3 cycle 1.

Next change I’d like to make is to change Step 1 requirement 2: Candle must be entirely below (when sd is +1) the BOS or CTS reference price, and it must be prior to the anchor candle of the pattern that breaks the reference BOS or CTS. For this requirement, I think the specs I gave were not correct. Hypothetically, let’s say for a particular structure that just reversed OR is in the counter direction from the parent sd, and we allow for cross fibs from cycles 0-3 in the reversed structure. As currently implemented, if a true cross fib exists from 0-3, then the only requirement 2 reference price we would have is the BOS reference because our current interpretation is that the entire 0-3 is a unified cycle that does not have a previous CTS to break. If the cross fib is 1-3, then cycle_0 CTS would be the CTS reference. If cross fib is 2-3, then cycle_1 CTS would be the CTS reference, and so on. However, this is actually the incorrect implementation.

In reality, regardless if we have cross fib 0-3, 1-3, or 2-3, the true CTS reference in all these cases should actually be cycle_2 CTS. From a breaking previous CTS perspective, we do not treat the cross fib cycles as unified. Similarly, if we have cross fib 0-1, cycle_0 CTS would be the CTS reference; if we have cross fib 1-2, cycle_1 CTS would be the CTS reference. The CTS reference will always be the max cycle-1 CTS. This now becomes consistent with single cycle fibs, where we are always using cycle-1 CTS as a reference. This also means that in essence the only instance where we would not have a CTS reference price would be when we are currently active on cycle 0. What this also means I think is that with regards to identifying the CTS reference price for requirement 2, we don’t need to treat single & cross fibs differently since the CTS reference should behave the same way every time we form a new cycle regardless whether it’s a new valid cross or single fib. This could then help simplify the code and implementation. That’s my guess here but you would be the engineering expert to make that judgement.

I think what this change would do is change the final IC for the sub_3 cycle 1-2 cross fib. Whereas previously the CTS reference was set to CTS_0, the new reference should be CTS_1, which would then extend the Step 1 requirement 2 threshold and unlock many more IC candidates.

Finally, some downstream housekeeping items. The new updated flow should look something like this:

Step 1 Identify IC candidates:
1. Candle must be in opposite direction of MS
2. Candle must be entirely below (when sd is +1) the BOS or CTS reference price, and it must be prior to the anchor candle of the pattern that breaks the reference BOS or CTS
3. There must be immediate unfilled imbalance after the candle, within IC + 2 (<=)
4. The candle must overlap with the 61.8 - 80 Fib zone

Step 2 Identify Institutional Pattern
Using qualified IC candidates from Step1, look for the highest ranked available Institutional Pattern that qualified IC candidates are a part of. If multiple equally ranked IP are found and multiple IC candidates remain, we apply tiebreakers:
1. Prefer maru > normal > pinbar
2. Prefer the most recent IC prior to the oldest (earliest) unfilled imbalance that comes after the oldest IC candidate

Step 3 is then processed if Step 2 cannot find any Institutional Patterns. Take all the IC candidates from Step 1 and apply a general ranking logic on IC candidates:
1. Exclude pinbar IC candidates
2. Prefer maru > normal
3. Prefer the most recent IC prior to the oldest (earliest) unfilled imbalance that comes after the oldest IC candidate

Step 4 Process Steps 1-3 on lower TF

Since the immediate unfilled imbalance rule moves up to Step 1, they are no longer needed in Steps 2 & 3, which shorten those rulesets. A potential consequence of the 2 Step 1 requirement changes is that now we produce fewer qualified IC candidates, which in the current flow shown below, could mean higher cases of reaching Step 4 (either from Step 1 directly or through Step 2-3), which is more complex.

```
Step 1 Identify IC candidates
├── IC candidates identified
│      └── Step 2 Identify Institutional Pattern
│              ├── final IC identified
│              └── Step 3 Apply general ranking logic on IC candidates when no IP exists
│                      ├── final IC identified
│                      └── Step 4 Process IC on lower TF when general ranking logic filters out all IC candidates
└── No qualified IC candidates
        └── Step 4 Process IC on lower TF
```

Therefore, I’d like to also propose a 3rd change that would add another step, effectively as 1A (let me know if you feel like something else would be more appropriate?)

Step 1A will be run only if Step 1 returns no candidates. In that case, then for Step 1 requirement 4, we change it to: The candle must overlap with the 80 - 100 Fib zone. Essentially, if we can’t find any candidates in the 61.8 - 80 zone, but there are candidates that meet all the other Step 1 requirements in the 80 - 100 zone, we can use those instead. From then on, we apply Steps 2-3 but now effectively to only candidates from the 80-100 zone, since no candidates exist from the 61.8-80 zone. This is why I also noted in the previous sessions that it is important the zone overlap rule exists outside of the institutional pattern logic so that it makes these processes & changes easier to implement. So now the overall flow will look something like this:

```
Step 1 Identify IC candidates
├── IC candidates identified
│      └── Step 2 Identify Institutional Pattern
│              ├── final IC identified
│              └── Step 3 Apply general ranking logic on IC candidates when no IP exists
│                      ├── final IC identified
│                      └── Step 4 Process IC on lower TF when general ranking logic filters out all IC candidates
└── No qualified IC candidates
        └── Step 1A Expand Fib levels overlap zone to 80 - 100 (but preserving requirements 1-3)
                ├── IC candidates identified
                │      └── Step 2 Identify Institutional Pattern
                │              ├── final IC identified
                │              └── Step 3 Apply general ranking logic on IC candidates when no IP exists
                │                      ├── final IC identified
                │                      └── Step 4 Process IC on lower TF when general ranking logic filters out all IC candidates
                └── No qualified IC candidates
                        └── Step 4 Process IC on lower TF
```

I name this new step 1A because in effect, it’s somewhat of an exception handling specifically when Step 1 in its original form returns no qualified candidates from the get go. It is not something applicable further downstream. So for example, let’s say we do have Step 1 IC candidates, but no IP and all candidates are pinbars, which would fail Steps 2-3. If that was the result, we would not go expand the fib zone to 80-100 and look for candidates there and re-run Steps 2-3. Thus, it would only ever come into play immediate after Step1.

I think changes 1 & 2 may be the easier ones to make, while the last change may be more complex (and could be deferred if needed), but curious to hear your thoughts there. Also love to just hear your thoughts in general on these changes. Do they make sense in terms of better identifying the IC? Any questions you have? Anything you’d like to clarify or discuss? Anything I said that is not correct or false? Any potential issues with implementing these changes or something that could cause conflict or confusion?
Let’s discuss.
