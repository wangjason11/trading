# The user's write-up (2026-10-20, session-order item 4 — the IC identification revisit), VERBATIM

Pasted by the user at the start of the IC-revisit session (baseline S_R3 `20261020_225749_d424099`, HEAD `896db93`),
for review and discussion before any plan. Two changes: Step 1B (a preference for IC candidates whose closest prior
imbalance is unfilled as of the IC) and the imbalance fill threshold 70 % → 65 %. The user's framing: "Please review
the details carefully and be prepared to discuss. Do not just take anything I say as true, please help double check
any assumptions I make." Kept verbatim (the rationale is the canonical record of WHY); the Q&A and measurements live
in `qa_findings.md` (this folder), decisions in the owning plan.

---

I want to add this preference rule in the IC selection process: prefer ICs where the closest imbalance prior to IC is still unfilled as of IC. What that means is that it does not matter if the imbalance becomes filled after the IC, as long as it’s still unfilled as of the IC. We also only care about the closest imbalance prior to the IC specifically. Any other prior imbalance doesn’t matter. 

This condition should be effectively added as Step 1B, and also very importantly is a preference condition rather than a requirement condition. Thus, it is applied after the Step 1 required conditions. Then amongst these Step 1 candidates, we prefer ones with unfilled imbalance prior to the IC. This is similar to the immediate unfilled imbalance after IC requirement from Step 1 but now applied to prior to the IC and not immediate, but only the closest imbalance. Candidates that meet 1B will always be preferable to ones that do not and if there are any available, they will move onto Step 2 (and if only 1 meeting 1B then that will be the final IC). However, if there are no candidates that meet this preferable condition, then we simply move on with the original Step 1 candidates to Step 2. The reason why this step is named 1B, is that in the event we have no Step 1 candidate, which would trigger Step 1A, Step 1B would be applied after 1A in the same way after Step 1. So amongst Step 1A candidates, we prefer ones with prior unfilled imbalance up to IC over all other candidates without it (if only 1, then that’s the final IC) to move onto Step 2. However, if none of the 1A candidates can meet the 1B condition, we continue onto Step 2 with the full 1A candidates. 

The flow should be like this:

After Step 1 Identify IC candidates 4 rules have filtered out our IC candidates, if we have IC candidates identified from Step 1, then we apply Step 1B to find IC’s with the closest imbalance prior to the IC still unfilled as of IC. The remaining candidates will then move on to Step 2 Identify Institutional Pattern. However, if there are no ICs that meet Step 1B condition, then all available candidates from Step 1 will move onto Step 2. 

On the other hand, when Step 1 results in no IC candidates, then we proceed as usual to Step 1A Expand Fib levels to find IC candidates. At this point, if 1A does identify candidates, then we apply 1B to those candidates with the same flow as above after (Step 2 on candidates that pass 1B OR if no candidates pass 1B, then all candidates from 1A move onto Step 2). If we are on Step 1A and we do not identify any qualified candidates after expanding the Fib zone, then we continue to Step 4 (same as current logic). 

At any point if 1B results in just 1 preferred candidate, then that’s the final IC.

```
Step 1 Identify IC candidates
├── IC candidates identified
│      └── Step 1B Prefer ICs with closest prior imbalance unfilled as of the IC
│              ├── IC candidates identified
│              │      └── Step 2 Identify Institutional Pattern
│              │              ├── final IC identified
│              │              └── Step 3 Apply general ranking logic on IC candidates when no IP exists
│              │                      ├── final IC identified 
│              │                      └── Step 4 Process IC on lower TF when general ranking logic filters out all IC candidates
│              └── No qualified IC candidates from Step 1B, proceed with Step 1 candidates
│                     └── Step 2 Identify Institutional Pattern
│                             ├── final IC identified
│                             └── Step 3 Apply general ranking logic on IC candidates when no IP exists
│                                      ├── final IC identified 
│                                      └── Step 4 Process IC on lower TF when general ranking logic filters out all IC candidates
└── No qualified IC candidates
        └── Step 1A Expand Fib levels overlap zone to 80 - 100 (but preserving requirements 1-3)
                ├── IC candidates identified
                │      └── Step 1B Prefer ICs with closest prior imbalance unfilled as of the IC
                │              ├── IC candidates identified
                │              │      └── Step 2 Identify Institutional Pattern
                │              │              ├── final IC identified
                │              │              └── Step 3 Apply general ranking logic on IC candidates when no IP exists
                │              │                      ├── final IC identified 
                │              │                      └── Step 4 Process IC on lower TF when general ranking logic filters out all IC candidates
                │              └── No qualified IC candidates from Step 1B, proceed with Step 1A candidates
                │                      └── Step 2 Identify Institutional Pattern
                │                              ├── final IC identified
                │                              └── Step 3 Apply general ranking logic on IC candidates when no IP exists
                │                                      ├── final IC identified 
                │                                      └── Step 4 Process IC on lower TF when general ranking logic filters out all IC candidates
                └── No qualified IC candidates
                        └── Step 4 Process IC on lower TF
```

Finally, coupled with the above change, the 2nd change I would also like to update our parameter that defines what % threshold price must move past to trigger eventual unfilled imbalance from 70% to 65%.

Now is there anything you’d like to clarify or discuss. Any nuances that I may have overlooked? Anything ambiguous? Any potential issues with implementation or conflicts or bugs that you can foresee? Any thoughts in general on this and if it is easy / feasible implementation? Any other additional details I can give?
