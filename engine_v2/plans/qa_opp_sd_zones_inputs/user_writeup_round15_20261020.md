# The user's write-up, round 15 — the gap thresholds gate reference UPDATES instead of capping triggers (2026-10-20, verbatim)

After the step-2 chart look (PLAN_R §8; step 2 built, not committed). Kept verbatim. Q&A: `qa_findings.md` round 15.

---

Thank you. I've reviewed the charts, and have a better understanding now. Assuming everything's implemented correctly, I would like to now pivot slightly. I see now with sub_8 issues where the opp_sd causes the gap to be very narrow and invert, and I think that was a point you were trying to bring up to me previously. Now that I can actually perceive the results, I think what makes more sense is rather than setting the gap thresholds as bounds where we would limit proximity triggers, we use those thresholds to limit what zones can be used as proximity references.

Let's say currently we are not too narrow. If the next update (on which ever side) causes the gap to fall below the threshold and become too narrow, then we do not make that update and the current reference zone remains what it was previously. We only make updates when the updated gap remains above the given threshold for each TF. Everything else with regards to what zones can be used on each side, etc. still remains the same.

This should actually simplify other things, specifically not needing to add another rule to enforce a single trigger per side. Instead, we always keep reference zones where we allow for unlimited alternations. The one exception to this is when the inner of the parent CTS & BOS themselves do not meet thresholds, since those inners delineate the widest the gap can be. If this was the case, this should be known as soon as CTS is confirmed. If this is the case, then that's the only place where we would limit alternations. Otherwise, as long as parent BOS to CTS inners is above threshold, then regardless of what other zones form, those will always at least be wide enough to serve as references to allow for unlimited alternations.

So previously on each potential update, we:
1. Identify / update references (starts initially with the inner of closest zones) per cycle, updates to next closest based on inner
2. Judge distance
    a. If too narrow: allow at most one sd and one opp_sd trigger per cycle after the update causes gap to go below threshold
    b. If not too narrow: no limit on alternation
3. Reset counter for updated side each time the gap expands and if the gap was below threshold prior to the expansion (The counter only matters when gap is below threshold prior to the update because that's when we limit the number of sd & opp_sd proximity triggers to 1. If the gap was already above threshold prior to expansion there would be no need to reset the counter since we do not limit triggers)

Now we do this:
1. Judge distance IF reference update were to happen (whether it is reduction or expansion)
    a. If too narrow: skip update
    b. If not too narrow: do update

There's no more counter reset needed either in the process.

What are your thoughts about this?
