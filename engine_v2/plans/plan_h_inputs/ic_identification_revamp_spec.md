# The holistic IC identification process — the user's write-up (verbatim, 2026-10-09)

> **Superseded in part by PLAN_P (2026-10-13):** the immediate-unfilled-imbalance rule left Steps 2 and 3 (it is
> IC condition 3 now) and Step 1A (the 80–100 fallback zone) was added — the user's write-up
> `../qa_ic_crossfib_inputs/user_writeup_20261013.md`, `../PLAN_P_ic_condition3_step1a.md`. The text below stays verbatim.

Item 1 (one POI per cycle), the IC identification revamp: the whole IC identification flow (Step 1 the IC
candidates, Step 2 the institutional pattern, Step 3 the general ranking when no pattern exists, Step 4 the lower-TF
fallback — deferred) and the two steps not yet built (3, 4). Received at `92b8b61` (baseline S_TBU
`20261009_171255_5b5c307`). This file is the user's text, unedited; the engineering reading, the measurements and
every decision live in PLAN_H (§6, §8.9).

---

Ok now that I’m happy with the institutional pattern logic, we can move onto the write-up of the holistic IC identification process along with the remaining parts. Here’s the concise summarized full flow:

Step 1. Identify qualifying IC candidates 
Step 2. Identify highest ranked Institutional Pattern (IP) from IC candidates 
Step 3. Apply general ranking logic on IC candidates when no IP exists 
Step 4. At any point, when we have no more qualified IC candidates, the fallback would be moving down to a lower TF and in essence run Steps 1 - 3 again to try and find the IC on the lower TF

As you recall we’ve now finalized the process to initially identify qualified IC candidates (Step 1). We then also finalized the Institutional Pattern logic to find the best IC amongst the qualified IC candidates (Step 2). However, presently if we cannot find an IP we simply do not have an IC. So to complete the IC identification process, we will now discuss the final steps (3 & 4) which are effectively exception handling & fallbacks when we don’t have IP or any qualified IC candidates remaining. Below are the detailed steps (with a lot of rehash). 

Step 1 we basically just finalized in our previous sessions where we apply 4 requirements / conditions to filter for IC candidates:
1. Candle must be in opposite direction of MS
2. Candle must be entirely below (when sd is +1) the BOS or CTS reference price, and it must be prior to the anchor candle of the pattern that breaks the reference BOS or CTS
3. There must be unfilled imbalance after the candle
4. The candle must overlap with the 61.8 - 80 Fib zone

Step 2 then runs when we have qualified IC candidates from Step1. In Step 2, we look for the highest ranked available Institutional Pattern that qualified IC candidates are a part of. If multiple equally ranked IP are found and multiple IC candidates remain, we apply tiebreakers:
1. Prefer IC immediately followed by unfilled imbalance, within IC + 2 (<=)
2. Prefer maru > normal > pinbar
3. Prefer the most recent IC prior to the oldest (earliest) unfilled imbalance that comes after the oldest IC candidate

Step 3 is then processed if Step 2 cannot find any Institutional Patterns. Step 3 would then take all the IC candidates from Step 1 and apply a general ranking logic on IC candidates. These general ranking rules will actually look very similar to the IP tiebreakers:
1. Exclude pinbar IC candidates
2. Prefer IC immediately followed by unfilled imbalance, within IC + 2 (<=)
3. Prefer maru > normal
4. Prefer the most recent IC prior to the oldest (earliest) unfilled imbalance that comes after the oldest IC candidate

Basically, we exclude all pinbars from the very start, then apply the other tiebreakers on the remaining IC candidates the same way as we did in IP. We generally do not want pinbar candles as our IC as they do not make strong POI zones. However, we allow it in the institutional pattern because the pattern indicates strength. Without IP, we exclude them entirely. 

Finally, if Step 3 fails to output any final ICs. The only way that this should happen is if all IC candidates are pinbar and so no candidates move down to rules 2-4 on the general ranking. We then reach the final fallback method to find the IC, which is essentially the last resort. We would need to go to a lower TF’s data, and repeat Steps 1 - 3 to find the IC there. This is actually something I want to defer for now because it could become quite complex, but may not give us much benefit if most of our cycles are able to find an IC using Steps 1-3. 

One last important thing in terms of process is that when Step 1 fails, or we cannot find any qualified IC candidates at all based on the 4 conditions, then we immediately jump to Step 4. Step 4 is meant to be the fallback at any time when we no longer have IC candidates, which is the case if no candles pass Step 1. So the full flow looks more something like this:

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

Logically, if Step 1 produces no IC candidates, then Step 2 cannot run because IPs form around IC candidates. In that case, we skip Steps 2 & 3 and jump directly to Step 4, our fallback when we don’t have any IC candidates. Step 3 also only runs if Step 2 runs AND Step 2 fails to produce a final IC, which would only happen if there are no IPs. Step 3 conditions will either filter out all IC candidates due to its rule 1 or produce a final IC by rule 4. In the case where all IC candidates are filtered out, we then fallback to Step 4. Step 4 again we will defer for now. We may also ultimately choose to skip Step 4 entirely and decide that not every cycle MUST have an IC & POI zone.

I think that’s the gist of the entire process to identify IC. Do you have any questions or confusions? Anything you want to clarify on or discuss? Any potential issues or conflicts you see? Any nuances I may have missed? Anything else to add? Open for discussion
