# SESSION ORDER item 5 — chart updates & edits: WVMI back on the chart (the user, 2026-10-24, verbatim)

Context: item 5's second pass (the charts' other elements), after the replacement break on every lens (`56517e7`, save
S_repl_break `20261024_171443_56517e7` = the `/compare` baseline). Register §G "WVMI back on the chart" — TABLED
2026-10-23, un-tabled here.

## The user's request (verbatim)

> Thanks. Next I'd like to work on adding WVMI back. Do you think we should continue in the same session or start a
> new session?

> let's continue here, go ahead

## Background (read 2026-10-24 from git, the specs and S_repl_break `20261024_171443_56517e7`)

- **What "on the chart" was:** only the wave-candle HOVER (added `cc60718`, removed `e64556a` 2026-05-28 — "Chart
  simplifications: drop momentum hover + Weighted vol"): every wave-candle line showed `Weighted vol: <volume × weight>`;
  an LB or LP line added `<Buy|Sell> momentum` (LB: `breakout_momentum`, LP: `pullback_momentum`) + the paired weighted
  volumes, looked up by candle INDEX (`wvmi_by_idx`). No marker, label or panel was ever drawn; WAVE_CANDLES_SPEC
  "Rendering simplifications": "Full WVMI momentum markers deferred (revisit later)"; PART4 §16.5's WVMI row: "locked
  records show final values via hover; in-progress records re-render whenever `update_temporary_lp` shifts the temp LP".
- **What WVMI is** (WVMI_SPEC): per cycle with a CTS_CONFIRMED (the main: only the cycles with a first sd proximity
  trigger; a sub: every cycle) — `breakout_momentum = LB_vol × LB_weight / FB_vol` (fixed at the CTS confirmation),
  `pullback_momentum = LP_vol × LP_weight / FP_vol` with a TEMPORARY LP (the qualified candle whose close is closest to
  the BOS zone's outer bound) until the next BOS locks it to that BOS's last wave candle; weights 1.0 / 0.7 / 0.5;
  `buy_momentum` / `sell_momentum` = the two mapped by zone side. Exports: `*_wvmi.csv` (H1) and
  `*_M15_{lens}_wvmi.csv`.
- **Records on S_repl_break:** H1 3 (sid 0 c1, sid 1 c1 collapsed, sid 1 c2); counter 7; confluence 18.
- **Role candles vs the drawn lines** (`wvmi_lines.py`, this folder — same owner + cycle + role):
  - every LOCKED, non-collapsed record: all four roles on drawn lines;
  - every OPEN (`created`) record: FB / LB / FP drawn, the temporary LP NEVER a drawn line of its own — on confluence
    it lands on ANOTHER sub's line twice (sub 3 c2's LP 3299 = sub 4's LB; sub 5 c0's LP 3773 = sub 7's
    pre-structure LP line), elsewhere on no line (H1 896, 1055);
  - the 6 records of COLLAPSED cycles (H1 sid 1 c1; sub 3 c0; confluence sub 0 c0, sub 1 c0 / c1): no line drawn;
  - sub 5 c0's FB 3300: not drawn as sub 5's own line on either lens (confluence: sub 4's FP sits there).
  So an index-only lookup (the old hover) shows another structure's numbers on a line, and an open cycle's pullback
  momentum has no line to sit on.

## Options put (2026-10-24) and the user's answer

Put: 1 hover only, keyed by owner + cycle + role (not idx); 2 = 1 + a marker at the temporary LP candle (the open
cycle's pullback momentum gets its own hover point); 3 visible labels (`BO 1.10` / `PB 0.76`); 4 a WVMI strip in the
volume area (each cycle's momentum over its candles). Scope: which charts; collapsed cycles hidden?

> Hmm actually I think I want to compare how 2 would look vs. 4.
>
> Scope
> yes all
> yes stay hidden

So: mock-ups of 2 and 4, on every chart (the H1 chart, the M15 sub lines, the M15 charts' H1 overlay); collapsed
cycles' WVMI hidden like their wave candles.

## The mock-ups (2026-10-24, `wvmi_mockups.py` this folder — drawn onto copies of S_repl_break's charts, no engine change)

- Option 2: the wave-candle hovers of every non-collapsed record's drawn lines gain a `WVMI` block, matched by owner +
  cycle + role — FB / LB the breakout momentum, FP / LP the pullback (an LP line only for a LOCKED record whose LP is
  that line); an open record's temporary LP gets a hollow purple diamond at its candle's close (H1 2 on every chart;
  counter 3, confluence 7 sub ones). Hovers extended: H1 6; counter 20 sub + 6 H1; confluence 48 sub + 6 H1.
- Option 4: a log-scale WVMI strip (0.05 – 12, dotted 1.0) between price (domain 0.33 – 1) and volume (0 – 0.15):
  per record a breakout segment FB → LB and a pullback segment FP → LP at their momentum, coloured by the wave's
  direction (buy green / sell red), H1 solid, M15 dashed, an open LP end hollow; hover = the record's values.
  Segments: H1 4; counter 12 + 4; confluence 28 + 4.
- Draft fields (both): `breakout_momentum=x (buy|sell) | fb_idx | lb_idx`, `fb_volume | lb_volume | lb_weight`;
  `pullback_momentum=x (…) | fp_idx | lp_idx`, `fp_volume | lp_volume | lp_weight`, `lp=locked (by cycle N's BOS)` /
  `lp=temporary (moves until the next BOS locks it) | status=created`. They repeat values a wave-candle hover already
  shows (`volume=`, `idx=`) — to settle with the chosen option. Volumes from the frames, weights back-solved (the CSV
  exports neither; the build reads them off `WVMIRecord`).

## Round 2 — the user's review of the mock-ups (2026-10-25, verbatim)

> Ok thank you for working on these mocks. Neither of them are exactly what I want, but from reviewing these, I now
> have a better understanding of what I want to show on the charts, so thank you for that.
>
> First, we still do not have the WVMI metric back that we previously did, which should be a function of breakout
> momentum (bm) and pullback momentum (pm). I believe it was bm / pm? But let's double check. That's the ultimate
> metric that really matters. WVMI starts mattering when both bm and pm are available, and the point that happens would
> be when CTS is confirmed, either through pullback pattern or proximity trigger. Once CTS is confirmed, then that
> guarantees that bm would be locked, and at this point we should also have the FP candle available. The final piece is
> the LP, and as should be previously mentioned in docs & specs, in a live trading setting, the LP would move as the
> price pulls back, eventually ending at the last locked LP candle. So what I think is valuable, is being able to track
> that WVMI change as the pullback approaches our zones. Ultimately, we may use WVMI before it even becomes fully locked
> to make trade decisions by judging the momentum of the breakout wave vs. the pullback wave when the pullback wave is
> near or within the zones to see if we want to enter in the MS direction, expecting the price to bounce. WVMI>1 would
> mean breakout momentum is stronger than pullback momentum and strong signal to enter, and vice versa when WVMI<1.
>
> So here's what I would want.
>
> 1. We don't need the WVMI temporary LP markers on the chart. Think of it more like this, effectively every single
>    qualified candle in any moment in time could be the temporary LP. So, every candle as of the CTS confirmed (pattern
>    moment for pullback, or proximity trigger candle) and after that can qualify as a LP is a temporary LP. WVMI should
>    then be calculated as of that candle. Then as we iterate through candles and MS, when we come to the next candle
>    that can qualify as an LP, we again calculate an updated WVMI as of that candle. The WVMI for any candles in
>    between the previous temporary LP and the current temporary LP will not change until a new LP updates the value.
>    So essentially, every candle after CTS confirmed can have an WVMI indicator as of that candle.
> 2. Option 4 creates an extra subchart for WVMI, which is not what we would want
> 3. I don't think I want any chart elements added for WVMI (the current Wave candle lines are enough) on the candle
>    charts either. Instead, I want a marker to exist on the volume subchart above each volume bar for each qualified
>    temporary LP on the pullback wave. These should be small dots located above the volume bars. Additionally there
>    could also be potentially 2 dots per volume bar. One would be green and for when WVMI is measured in the buy
>    direction (BOS is buy zone), and the other one would be red for when WVMI is measured in the sell direction (BOS
>    is sell zone). It is possible that the same candle can have both a buy & a sell WVMI due to there being a both a
>    live/active counter & confluence relative_dir sub on the same candle, hence why we would have up to 2 dotes. The
>    green dot should be slightly above the red one.
> 4. LP marker data hover:
>
>    Sub / TF=x / lp_idx=x / breakout_mom=x / pullback_mom=x / wvmi=x / wvmi_dir=buy/sell / status= /
>    relative_dir=confluence/counter (one field per line in the original)
>
>    Now when we get to the locked LP wave candle, or at idx where we have the LP vertical line, we would still have a
>    dot above the volume bar. But this dot's hover will be differentiated from the other temporary LPs with the status
>    label, where it will be noted as "locked". We can discuss further what other potential statuses that can go
>    there.
> 5. Lastly we would also come into the scenario where the 1H main data would overlay on the sub charts. The hover data
>    would look similar, but slightly different:
>
>    Main / TF=x / lp_idx_1H=x (this row is only needed on the sub chart when main data overlays, on the main chart
>    itself we would just use lp_idx) / lp_idx=x / breakout_mom=x / pullback_mom=x / wvmi=x / wvmi_dir=buy/sell /
>    status= (one field per line in the original)
>
>    Additionally, for main charting, instead of using dots, I would like them to be circles. The color logic should be
>    exactly the same. The positioning should also be the same as subs above the volume bars. In the case where the
>    same sub candle has both sub & parent LP marker, the parent circle should fit neatly around the sub dot when
>    overlapping. Thus this allows us to use space efficiently without needing to further stack 1H markers as separate
>    rows. Does this description make sense to you in terms of what I want?
>
> I think those are the specs I have in mind for now. You probably have many questions and points you want to clarify
> so let's discuss. What do you think?

Why it matters: WVMI becomes a live entry signal (the breakout wave's momentum vs the pullback wave's, read while the
pullback nears the zones), so the chart must show its value AS OF each candle, not only the end-of-run record.

### Measured for the discussion (2026-10-25, S_repl_break code; `wvmi_asof.py` in the session scratchpad — a read-only wrapper on `orchestrator._compute_wvmi_records` replaying `_find_temporary_lp` with the search end = t)

- **No combined WVMI number has ever existed** in this repo: since `cc60718` (2026-02-13) WVMI = `breakout_momentum`,
  `pullback_momentum` and their direction-labelled copies `buy_momentum` / `sell_momentum` (WVMI_SPEC "Direction
  Labels"); the old hover showed `<Buy|Sell> momentum`. `wvmi = bm / pm` would be new. Check: the records' final
  bm / pm = the last as-of value (H1 (0,1) 1.102 / 0.759 = 1.45; H1 (1,2) 0.214 / 0.082 = 2.61).
- **The engine's LP rule replayed as of each candle** (the closest-to-outer qualified close in `[FP + 1, t]`, ties to
  the earlier) changes the LP only when a NEW qualified candle closes deeper, and then the new LP IS candle t —
  except the first value at the CTS_CONFIRMED moment, which can be an earlier candle: visible case confluence +
  counter sub 8 cycle 0, moment 4103, LP 4102 (also the collapsed confluence sub 0 cycle 0, 802 → 801).
- **Dots per chart** (non-collapsed records): this rule H1 42, counter 74, confluence 158; "every latest qualified
  candle" H1 185, counter 508, confluence 819. Example, confluence sub 3 cycle 2 (sell, CTS confirmed 3086 =
  2026-01-02 01:30): this rule 27 dots (3086 4.57, 3112 1.96, 3118 4.27, 3123 5.64, …); the latest rule 203, e.g.
  3100 (close 0.57616, moving AWAY from the zone) wvmi 11.73.
- **The locked LP (BOS_{n+1}'s last wave candle) = the last temporary LP** on every visible record; the one exception
  is collapsed (confluence sub 1 cycle 0: last temporary 1866, locked 1837).
- **No two dots share a candle** on either M15 chart (no same-colour, no buy + sell pair) on this window. H1 circles
  at the hour's last M15 candle (`4h + 3`) share a candle with a same-colour sub dot at 3755 (both charts) and 3499
  (confluence).
- **The H1 gate**: H1 records exist only for cycles with a first sd proximity trigger (5 CTS_CONFIRMED cycles → 3
  records, 1 collapsed). H1 (0,1)'s CTS confirmed at 683 but its gate trigger fired at 748.
- How the visible records end: H1 — a reversal (0,1), open at the data end (1,2); confluence (14) — locked by
  BOS_{n+1} 7 (sub 0 c1, 0 c3, 1 c3, 2 c0, 3 c1, 8 c0, 8 c1), a reversal 5 (0 c4, 1 c4, 2 c1, 4 c0, 7 c0), the sub
  replaced 2 (3 c2, 5 c0); counter (6) — locked 3 (3 c1, 8 c0, 8 c1), the sub replaced 3 (3 c2, 5 c0, 6 c0).

## Round 3 — the user's answers (2026-10-25, verbatim)

> Thank you for checking on "The formula: checked, and it never existed as one number".
>
> I believe I misremembered then, and WVMI should not be a separate metric that equals bm / pm. Actually breakout_mom
> and pullback_mom do need to exist on their own and the idea WVMI should not be trying to combine them The correct way
> to use these metrics would be to look at all the different possible combinations, which are:
>
> breakout_mom > 1 + pullback_mom > 1
> breakout_mom > 1 + pullback_mom <= 1 (ideal)
> breakout_mom <= 1 + pullback_mom > 1 (least ideal)
> breakout_mom <= 1 + pullback_mom <= 1
>
> Here you can see that when breakout momentum is increasing while pullback momentum is flat or decreasing, that's the
> ideal indicator for a high chance that price will rebound from a zone. When breakout momentum is flat or decreasing
> while pullback momentum is increasing, that's an indicator that there is a good chance the price will continue to
> pullback and maybe pass through the zone or potentially even hit a reversal in the future. For now this is all
> theory, down the line we will build this logic in more detail when we get to entry and other decisions & further
> test the validity of these claims. For now, this is just a bit of a preview for you and to explain better how we
> would use WVMI which is not doing bm / pm like I previously stated.
>
> So given that, we can remove wvmi as a hover label.
>
> Q1. Here I think it depends on our definition of qualifying candle. Can you look that up to see if we have defined
> it? I think in the past I may have already mentioned something along the lines of qualifying candle must close
> further in the pullback direction, but I'm not sure
>
> Q2. Hmm on this can you provide some more information on the method for CTS confirmed and why the confirmed happens
> on 4103?
>
> Q3. sounds good
>
> Q4. Hmm sorry can you give me more details on what this gate is you are referring to? What do you mean by as subs did
> in week 8? So you mean the gate is already gone? But also what do you mean it reverses the proximity gate? It's
> unclear to me right now whether that's something we would want to do without having better understanding of what you
> mean here by gate.
>
> Q5. Sounds good. Though these will need to be adjusted a bit given the above change I mentioned
>
> Q6. a) is good
>
> Q7. yes fixed row is good
>
> Q8. sounds good
>
> Any other questions or things to clarify and discuss?

Decided in this round: no combined `wvmi` value anywhere (hover or CSV) — WVMI = the pair `breakout_momentum` /
`pullback_momentum`, read by the four combinations around 1 (entry logic later, Week 9 Part 2+); statuses
`temporary` / `locked` / `open` / `unlocked` (Q3); the hover layout with `sub_id` / `cycle_id`, the CSV's full field
names, `lp_status`, 3 decimals, `relative_dir` at the LP candle (Q5, less `wvmi=`); an H1 LP on the M15 charts at
the hour's last M15 candle (Q6 a); fixed rows, red just above the bar, green one step higher (Q7); the per-LP rows
on the record + a new CSV per chart (Q8, less the `wvmi` column).

### Measured for round 3 (2026-10-25; `wvmi_asof.py` now also replays a "furthest close" rule)

- **"Qualified candle"** (GLOSSARY, WAVE_CANDLES_SPEC "Qualified Candle"): candle direction = the pullback direction
  AND the agreed volume direction (vol_dir the same or 0). Nothing about where it closes. WHERE it closes is the
  SELECTION rule: the qualified close **closest to the BOS zone's outer edge** (`abs(close - outer)`, WVMI_SPEC
  "Temporary LP Selection"; the wave candles' BOS Last Pullback searches use the same closest-to-outer rule). The
  phrase "further in the pullback direction" in the user's docs is from the proximity-reference write-up (round 4,
  2026-10-19), about sub zones as references.
- **Closest-to-outer vs furthest close in the pullback direction** agree until a close passes the outer edge. On
  S_repl_break they differ on 6 cycles, all ended by a reversal (none locked): H1 sid 0 c1 (outer 0.5736; the
  engine's LP stays 896 while 897 → 901 close 0.57334 → 0.57276), confluence sub 0 c4, 1 c4, 2 c1, 4 c0, 7 c0. Dots
  H1 42 vs 47, counter 74 vs 74, confluence 158 vs 185. WVMI measures to the zone's `meta["outer"]`, the outer at
  creation: 3 of the 6 zones later expanded (`expanded: True`; confluence sub 4 c0 outer 0.57418, drawn bottom
  0.57376). No locked cycle closed past its edge, so every locked LP = the last temporary LP under both rules.
- **Sub 8 cycle 0's CTS confirmation at 4103** (buy; cycle 0 — the proximity route is skipped on cycle 0, MS spec
  "Cycle-0 carve-out"): the pullback pattern `one_maru_opposite` (direction −1): c0 = 4102 (2026-01-16 15:30) a big
  bearish maru 0.57521 → 0.57444, body 74 %, closing below the range's low (the pattern's break check against the
  pullback threshold `range_lo`); c1 = 4103 (15:45) a small bullish pinbar (body 24 %, high 0.57502) whose high stays
  below that level (the pattern's pinbar path — its main path fails: c1 high 0.57502 ≥ c0 low + half its range
  0.57472). A two-candle pattern completes when c1 closes → the moment 4103; the deepest qualified close then is
  c0 = 4102. Every other visible cycle starts with the LP on the confirmation candle itself.
- **The H1 gate** = `_first_sd_prox_gate`: an H1 cycle gets a WVMI record only if its first sd zone-proximity trigger
  fires (`6c7f1a3`, 2026-02-16, Week 8 Part 2). Window: 5 H1 CTS confirmations → 3 records (sid 0 c1 confirmed 683 /
  trigger 748; sid 1 c1 810 / 820, collapsed; sid 1 c2 935 / 935); sid 0 c0 (439) and sid 1 c0 (727) had no sd
  trigger. Subs: gated by the parent's triggers until Plan G (2026-09-30) made them ungated. **The H1 records also
  start the M15 `first_counter` triggers** (`multitf/uc1_trigger.detect_uc1_triggers` iterates the H1 WVMI records;
  the H1 marker hover reads `sub_trigger=first_counter` at 748 / 820 / 935) — dropping the gate is not a WVMI-only
  change. My round-2 recommendation to drop it (made before I saw this) withdrawn.

## Round 4 — the user's answers (2026-10-26, verbatim)

> Q1. you mentioned that under the "furthest" definition, the LP would follow to 901. I'm wondering why would it stop
> at 901 instead of going to 902, which would have a lower close?
> Yes I agree with b) the previous definition we had did not account for the fact that price can go past the BOS zone,
> and when that happens, we continue to follow those candles and have a new LP each time we have a lower close. Again,
> this case I feel like that should extend to 902 and not just 901?
>
> Also to rehash, from a definition perspective, to be a "qualified" last pullback candle, the candle's direction must
> be in pullback direction and its volume direction agrees (or is neutral).
>
> Where the close lands you state is currently a separate rule. I think that's fine, but I think ultimately we can also
> combine it together as part of the "qualification" of being the last pullback candle. Since now we would track /
> update the LP as price moves & pulls back, the same close lands rule can be restated as: "(in addition to the
> direction rule above), the LP (or temp LP) must also close further in the pullback direction than the previous LP (or
> temp LP)."
>
> This then establishes how we find the very first LP (when CTS confirmed), and then how we would update LPs
> iteratively afterwards. What do you think about this? Should we keep the direction qualification & close logic
> separate? Or do you think it makes sense to combine them into 1 set of qualification conditions?
>
> Q2. I see. Thank you for the explanation. I would like to make a correction to the logic I had earlier: "Only
> candles from the confirmation onward can be LPs." Instead, I want to apply the following depending on how CTS was
> confirmed to identify the very first temporary LP:
> - via pullback: Any candle that's part of the confirmed pullback pattern can be the LP given that it is 1) the right
>   direction, and 2) closes closest toward the pullback direction. So this would cover the case where the lowest
>   candle happens not on the past candle of the pullback pattern
> - via proximity: when CTS is confirmed via proximity, at the proximity candle, if we have prior candles that are 1)
>   the right direction, and 2) close lower than the close of the proximity candle, we take the candle with the lowest
>   close amongst these prior candles. If no prior candle meets 1 & 2, then we see if the proximity candle itself is
>   the right direction; if so, then it is the first temp LP. Finally, if the proximity candle itself cannot be the
>   first LP, then we look (or wait) for the first LP after the proximity candle. The first candle after proximity that
>   is 1) in the right direction, and 2) closes closer toward pullback direction than proximity candle's close is our
>   first temp LP
>
> Now after finding the first temp LP, for all future LPs, we simply look for a candle that's 1) the right direction,
> and 2) closes closer than the previous temp LP's close in the pullback direction.
>
> Q4. Can you provide more explanation around "we would actually move WVMI directly to unique subs… Conceptually, this
> is the same as zones." How did moving the WVMI directly to unique subs specifically remove the previous gate? So now
> how is WVMI triggered on subs that is different?
>
> For option a), if we choose to start LP at CTS confirmation, what do you mean by "This shows values before the cycle
> qualified." What does qualified cycle mean?
>
> Based on your description, I'm leaning toward updating the gate to being CTS confirmed rather than proximity.
> Ultimately, in many cases those will be the same thing since CTS can be confirmed via proximity. But in other cases,
> it would allow WVMI to start earlier when CTS is confirmed via pullback before proximity triggering. But before
> anything let's discuss more. Would there be any issues with this?
>
> Finally, I'm also not sure what you mean by first_counter reads the proximity trigger directly, and how that is
> relevant to 1H chart? Can you explain that?
>
> Hover looks good
>
> We can settle on the one more thing after we've discussed my above questions. Other than these items are there
> anything else you want to bring and discuss?

Decided in this round: Q1 (b) — the LP follows the pullback past the BOS zone, a new LP at each further close; the
hover drafts of round 3 (no `wvmi`, statuses with their why / when). Proposed by the user, under discussion: fold
"closes further than the previous LP" into the LP's qualification; the first temporary LP chosen by how the CTS
confirmed (the pattern's candles / the proximity candle's rules); the H1 gate moved from the first sd proximity
trigger to the CTS confirmation.

### Measured for round 4 (2026-10-26; `wvmi_asof.py` now also replays the user's round-4 rules + an ungated H1, report `wvmi_r4_report.py`)

- **Why the furthest-close LP of H1 sid 0 c1 stops at 901, not 902:** 902 is the candle that ENDS the cycle — its
  close confirms sid 0's reversal (pattern `continuous`, anchored 897, confirmed 902: `STATE_CHANGED` pullback_range →
  reversal at 902), and at the same close sid 1's cycle 2 `CTS_ESTABLISHED` + `BOS_CONFIRMED` fire. The lifecycle is
  half-open `[652, 902)`, so the LP search ends at `end − 1` = 901 (WVMI_SPEC "Temporary LP Selection" 1). 902 itself
  is qualified (bearish, vol_dir −1) and closes lower (0.57242 vs 901's 0.57276). The same holds on ALL 6
  reversal-ended cycles: the ending candle is qualified and closes further (H1 902; confluence 1940, 2470, 2829, 3589,
  4200). Replacement endings (3491, 3819, 4143): qualified, not further. Lock candles (BOS_{n+1} confirmed): never
  qualified.
- **"Qualified candle" is a shared term:** WAVE_CANDLES_SPEC `_is_qualified` (direction == required_dir AND vol_dir
  agrees or 0) serves every wave-candle search (FB / LB / FP / LP, BOS and CTS zones). The close condition compares
  with a PREVIOUS LP, which only the LP has.
- **The user's first-LP rules (rule_r4) = the furthest-close rule on every record of this window:** dots H1 47,
  counter 74, confluence 185 (identical sequences). 16 visible confirmations by a pullback pattern (15 unique subs +
  H1 sid 0 c1), 1 by proximity (H1 sid 1 c2 at 935: itself qualified, no earlier qualified close beyond its 0.5772 →
  itself; the deepest earlier qualified close is 931 at 0.5768). Pattern route: no qualified candle BEFORE the pattern
  closes further than the pattern's deepest qualified candle (0 cases). 7 of the 16 patterns start ON FP (sub 0 c3,
  0 c4, 1 c4, 2 c0, 2 c1, 3 c1, 8 c1; e.g. sub 3 c1 pattern 2843–2844, FP 2843); FP never closes further than the
  first LP, and every pattern has a qualified candle after FP (the LP search starts at FP + 1 today).
- **The locked LP** (the wave candles' BOS_{n+1} Last Pullback — its own windows around the BOS_{n+1} anchor, closest
  to that zone's outer) = the last rule_r4 temporary LP on every visible locked record (7 sub records; 8 with an
  ungated H1's sid 0 c0, LP 590).
- **How Plan G removed the sub gate** (PLAN_G §1–§3): before it a sub had WVMI only when a parent-trigger stream entry
  of one of its lenses landed inside its window (subs 0–2: none → no WVMI); Plan G moved the computation into each
  unique sub's projection, beside its KL / POI / fib — every cycle with a CTS_CONFIRMED (and a CTS wave FP) gets a
  record; the triggers became attribution (`triggered_by_*` = the lens's first trigger inside the sub's window, blank
  if none). The user chose Q1 (A) "no gate on subs; the main keeps its first-sd gate" over (B) "none anywhere". The
  gate never changed a value (Plan G §2: 8/8 records identical), only a record's existence.
- **"Gate = CTS confirmed" = ungated** (every record is created at its cycle's CTS_CONFIRMED). Measured: H1 +2
  records — (0,0) buy, CTS confirmed 439 (pattern 436–437 + confirmation 439), FP 436, LP 590 locked, breakout
  momentum 0.733, 10 dots, visible; (1,0) sell, 727, COLLAPSED `[902, 902)` (sid 1 exists from the reversal 902; its
  cycles 0–1 are reconstructed then) → hidden. H1 dots 47 → 57 (furthest rule).
- **first_counter's coupling:** `multitf/uc1_trigger.detect_uc1_triggers` loops over the H1 WVMI records; the trigger
  moment = the record's `meta["triggered_by_event_idx"]` (the gate's first sd trigger). Window: (0,1) 748 → M15 2995
  → counter sub 3; (1,2) 935 → 3743 → counter sub 6; (1,1) 820 → 3283 unresolved (`collapsed_parent_cycle`).
  Ungated records (0,0) / (1,0) carry no trigger → `_run_multi_tf_dual`'s assert "has no trigger_event_idx". Readers
  of the H1 records: `_run_multi_tf_dual` (`detect_uc1_triggers`), the chart's `proximity_sub_trigger_types` (the H1
  cross hover's `sub_trigger`), the CSV export. `_final_proximity_triggers` asserts each cycle's first proximity
  trigger is unchanged by the sweep ("the main WVMI gate read it") — first_counter reads it too.

## Round 5 — the user's answers (2026-10-26, verbatim)

> why the LP stops at 901 and not 902. a) sounds good
> Q1. sounds good
> Q2. 3 things to confirm
> 1. Yes
> 2. b) never FP. yes we can apply something similar to proximity first LP logic that I gave above
> 3. Yes intended
>
> Q4. Sounds good
> 4 Locked LP. sounds good
>
> Anything else worth bringing up to discuss or clarify?

Decided in this round:
- **The LP search ends at the cycle's end − 1** (option a): the candle that ends a cycle (a reversal's, a
  replacement's, the next cycle's BOS) is never an LP — the same end as the cycle's zones and wave-candle lines.
- **One LP rule, "qualified" unchanged.** Qualified (every wave-candle role) = direction = the pullback direction AND
  the volume direction agrees or is neutral. The temporary LP rule (stated once, WVMI only) = a qualified candle
  closing further in the pullback direction than the current temporary LP; an equal close does not move it.
- **"The right direction" = fully qualified** (Q2 1).
- **Never FP** (Q2 2, option b): the LP search starts at FP + 1 on every route. The pattern route gets a fallback
  "similar to" the proximity route's first-LP logic (exact wording to confirm, round 5 reply).
- **The pattern route looks only at the pattern's own candles; the proximity route looks back to FP + 1** — the
  asymmetry is intended (Q2 3).
- **The H1 gate goes** (gate = the CTS confirmation = no gate, as subs since Plan G); first_counter is rewired to
  read the H1 cycles' first sd proximity triggers directly, landing first (Q4).
- **The locked LP**: keep the swap to the wave candles' BOS_{n+1} Last Pullback; the replay warns when it differs
  from the last temporary LP.

### Measured for round 5 (2026-10-26; from `wvmi_asof_r4.json` + the save's charts / CSVs)

- **The pullback pattern's span.** CTS_CONFIRMED's meta carries no pattern candles (its keys: confirmation_method,
  confirmed_at, cts_anchor_idx, cycle_id, structure_id, struct_direction, via, + proximity / sub keys);
  CTS_ESTABLISHED carries `pattern_anchor_idx` (the pattern's first candle). On all 17 pattern confirmations the CTS
  confirmation moment = the pattern's apply candle (its end, or its confirmation candle when CONFIRMED). A CONFIRMED
  pattern's confirmation candle is the first of up to 4 candles after the pattern's end that confirms
  (`patterns/structure_patterns._price_confirmation`), so candles can sit between the end and the confirmation:
  H1 sid 0 c1 680–681 / 682 (bullish pinbar, not qualified) / 683; the ungated H1 sid 0 c0 436–437 / 438 (bullish
  pinbar) / 439; confluence sub 7 c0 (sell) 4154–4155 / 4156 (qualified, close 0.57765) / 4157 (close 0.57785, the
  first LP). No in-between candle is the deepest on this window.
- **Never FP, the near miss:** confluence sub 1 c4 (sell): `one_maru_opposite` 2365–2366 starts ON FP 2365 (close
  0.57412); 2366 (the opposite candle) closes 0.57412; the confirmation candle 2367 (close 0.57452, qualified) is
  the only qualified pattern candle after FP → the first LP. Without a confirmation candle the pattern would have
  none → the fallback (the first later qualified close above 2366's 0.57412) = 2367 too.
- **The new LP rule's effect** (the final LP of every record): 6 unlocked records change, all ended by a reversal,
  each now ending on its end − 1 or near it — H1 sid 0 c1 896 → 901 (pullback momentum 0.759 → 0.478); confluence
  sub 0 c4 1898 → 1938 (0.881 → 0.585), sub 1 c4 2455 → 2469 (0.356 → 1.019), sub 2 c1 2734 → 2828 (0.657 →
  2.425), sub 4 c0 3568 → 3588 (0.817 → 1.299), sub 7 c0 4177 → 4196 (0.279 → 0.554). 3 of them cross 1 (≤ 1 →
  > 1: sub 1 c4, 2 c1, 4 c0). Every other record unchanged (H1 sid 1 c2 1055; counter sub 3 c2 3299, 5 c0 3773,
  6 c0 4002; confluence sub 3 c2 3299, 5 c0 3773; every locked record = its wave candles' LP).
- **Readers of a WVMI record's LP:** `debug/export_wvmi.py` (the CSV) and the orchestrator's `[wvmi]` count print
  only; no chart reads it (WAVE_CANDLES_SPEC: the temporary LP is "not drawn").
- **CTS_{N+1} ESTABLISHED and BOS_{N+1} CONFIRMED share the moment** on every cycle of this window (34 pairs: H1 5,
  counter 8, confluence 21), so WAVE_CANDLES_SPEC's "LP_N locks at the CTS_{N+1} ESTABLISHED moment" and WVMI's "at
  BOS_{N+1} CONFIRMED" name the same candle here (a wording difference to align in the docs).

## Round 6 — the user's answers (2026-10-26, verbatim)

> 1. I'm a bit confused by your confluence sub 7 example. The structure is in -1 direction, and when price retraces, it
> gets close to the POI first, so shouldn't that that confirm CTS via proximity instead of pullback pattern? The POI
> proximity would have happened much earlier like around 4134?
>
> But to answer your question broadly, yes a) is correct. We look at the full pattern from the anchor candle to the
> confirmation candle.
>
> 2. On the fallback, there is a correction to your logic. Instead of:
> "wait for the first later qualified candle that closes beyond the close of the candle that confirmed the CTS."
> It should be
> "wait for the first later qualified candle that closes beyond the close of the FP".
>
> It seems that the whole point of the fallback is that the FP is the only qualified candle, that's part of a pullback
> pattern, and so we do not allow it to be also used as an LP. That would also means all other candles in the pattern
> are not in the direction of the pullback, and should mean that the FP was the candle that should close closest in
> the direction of the pullback. Therefore, the LP close should be beyond that rather than the confirmed candle, is
> going to close away from the pullback direction.
>
> Otherwise looks good.
>
> 3. Ok
>
> 4. Your recommendation sounds good
>
> Small things. Yes the CTS established & BOS confirmed here shouldn't cause you any confusion. We have it well
> documented that from a moment perspective, those 2 are the exact same things.
>
> Anything else you'd like to bring up to clarify and discuss?

Decided in this round:
- **The pattern's candles = the full span from its anchor (first) candle to its confirmation candle**, in-between
  candles included (option a).
- **The pattern-route fallback** (no qualified pattern candle after FP): wait for the first later qualified candle
  closing beyond **FP's close** (the user's correction; not the confirmation candle's close).
- Accepted as stated: never FP on the proximity route too (p = FP → step 2 skipped, step 3 waits; step 1 looks from
  FP + 1 to p − 1); while waiting the record has no LP (no dot, empty `lp_idx`), a lock before any LP takes the wave
  candles' LP (+ the warning), any other end leaves the record without an LP.
- **Landing order** (the 5 steps of the round-5 reply, steps 4 and 5 separate): 1 first_counter reads the H1 first
  sd proximity triggers directly; 2 the H1 WVMI without the gate; 3 the new LP rule; 4 the per-LP rows + a new CSV
  per chart; 5 the dots + hovers on the three charts.
- CTS_{N+1} ESTABLISHED and BOS_{N+1} CONFIRMED are the same moment by definition (the user: "well documented").

### Measured for round 6 (2026-10-26)

- **Why confluence sub 7 cycle 0 confirmed by the pullback pattern at 4157, not by proximity:** it is CYCLE 0, and
  MS never confirms a cycle-0 CTS by proximity (MARKET_STRUCTURE_SPEC "Cycle-0 carve-out (proximity disabled)";
  LANDMINES "Dual CTS Proximity Confirmation — Cycle-0 + Rule 1 Are BOTH Required": BOS_0 = `initial_prior_extreme`,
  a weaker anchor "even when its gap happens to be wide" — both gates kept by the user's decision, 2026-05-12 Q3).
  Sub 7 c0 IS that wide-gap cycle 0: CTS 0.57255 (4000), BOS threshold 0.57827 → a 57.2-pip gap, past the M15
  narrow-gap floor of 30. Its sell POI 0.57653–0.57705 (ic 3809, confirmed 3819), BOS zone 0.57758–0.57898. The
  pullback's high first came within 5 pips (M15 `DEFAULT_PROXIMITY_PIPS`) of the POI's lower edge 0.57653 at 4071
  (high 0.57638), again at 4134 (0.57639), and touched it at 4138 (0.57653); within 5 pips of the BOS zone's inner
  0.57758 at 4141. The black cross at 4141 on the confluence chart is the H1 structure's own sd:POI proximity
  (H1 1035, parent sid 1 c2, `subsequent_counter`), not sub 7's.
- **FP's close vs the first temporary LP's close (rule_r4):** on all 17 visible records (18 with the ungated H1
  sid 0 c0) the first LP closes beyond FP, by 1.1 pips (sub 8 c1: FP 4158 0.57753, first LP 4159 0.57742) to 54.2
  pips (sub 4 c0); next narrowest sub 3 c1 +1.3 (FP 2843 0.57958, first LP 2844 0.57971).

## Round 7 — the user's answers (2026-10-26, verbatim)

> 1. let's go with b)
> 2. yes we do need to add line to revisit this. I need to better understand the original carve out logic. A decision
> from May was also quite a long time ago and a lot of things have changed since then. From your description of "a
> cycle-0 BOS is the swing extreme from before the structure began" that sounds like this was implemented
> specifically to cover the case for sid=0 cycle=0 on main TF. In all other cases, cycle0 form as a result of
> existing structure logic & triggers. Additionally, when we come to unique subs, there may be many subs that are just
> single cycles, and having the carve out logic forces all those to confirm CTS via pullback pattern. Those are my
> thoughts for now. We can come back to it in more detail later
>
> Are there any other outstanding items we need to decide or discuss?

Decided in this round:
- **FP's close is the bar for the pattern-route FALLBACK only** (option b): a first LP picked from the pattern's
  qualified candles, or by the proximity route, is not required to close beyond FP.
- **The cycle-0 carve-out is parked for a revisit** (register §G "Revisit the cycle-0 carve-out"; a pointer in
  MARKET_STRUCTURE_SPEC "Cycle-0 carve-out"). Unchanged until then; not part of this edit.

### Measured for round 7 (2026-10-26) — facts for the carve-out revisit

- **Every structure's cycle-0 BOS carries `source: "initial_prior_extreme"`** on this window: H1 2 (sid 0, and sid 1
  — the reversal's), counter 4, confluence 8 structures; every later cycle's BOS is `pullback_extreme`. The label is
  the same on all; how each structure's cycle-0 BOS is found differs (to lay out when the revisit starts).
- **Every cycle-0 CTS confirmation is by a pullback pattern** (H1 2, counter 4, confluence 8); by proximity: H1 1
  (sid 1 c2), confluence 1 (sub 1 c1, collapsed), both cycle > 0.
- **The two original cases** (LANDMINES, the entry above): H1 sid 0 c0 AND sid 1 c0 (a reversal's cycle 0, not only
  the main sid 0), both with a ~12-pip BOS–CTS gap — which the narrow-gap gate (H1 50 pips) now blocks on its own. So
  what the carve-out still blocks today is a cycle 0 with a WIDE gap, e.g. confluence sub 7 c0 (57 pips).
- Visible sub WVMI records on cycle 0: 6 of the 15 unique (subs 2, 4, 5, 6, 7, 8).

### The hover drafts the user accepted (round 3 reply → "Hover looks good", round 4)

```
TF=15M
Sub WVMI
sub_id=3 | cycle_id=2 | relative_dir=counter
lp_idx=3112 | lp_status=temporary (replaced by a deeper close at 3118)
wvmi_dir=sell | breakout_momentum=1.287 | pullback_momentum=0.658
```
```
TF=15M                      (on the H1 chart: TF=1H, and no lp_idx_1H)
Main WVMI
sid=0 | cycle_id=1
lp_idx_1H=896 | lp_idx=3587 | lp_status=unlocked (cycle ended by a reversal at 902)
wvmi_dir=buy | breakout_momentum=1.102 | pullback_momentum=0.759
```
- `as_of_idx` joins the `lp_idx` line only when it differs.
- Each status carries its why / when: `locked (cycle N+1's BOS confirmed at idx)`, `open (still moving at the data
  end)`, `unlocked (reversal / sub replaced / sub ended at idx)`, `temporary (replaced at idx)`.
- The example values are from the closest-to-outer rule (round 3); the new LP rule changes some (H1 sid 0 c1's final
  LP 896 → 901).

## Round 8 — the user's answers (2026-10-26, verbatim)

> 1. Sounds good
> 2. Let's use zone_side
> 3. Ok
>
> Anything else worth discussing?

Decided in this round:
- **The per-LP CSV** (step 4): `*_wvmi_lps.csv` (H1) and `*_M15_counter_wvmi_lps.csv` / `*_M15_confluence_wvmi_lps.csv`,
  one row per dot; columns = the record keys named as in `_wvmi.csv` (`structure_path_id`, `parent_sid`,
  `parent_cycle_id`, `sub_id`, `bos_structure_id`, `bos_cycle_id`), `relative_dir` (M15), `zone_side`, `lp_idx`,
  `as_of_idx`, `lp_status` / `lp_status_reason` / `lp_status_idx` (the hover joins the three), `breakout_momentum`,
  `pullback_momentum` (full precision; the hover rounds to 3).
- **The direction column is `zone_side`** (option b, the `_wvmi.csv` name), not `wvmi_dir`.
- **Status wording:** `temporary (replaced by a further close at idx)` (was "a deeper close"); a new reason
  `temporary (replaced by the locked LP at idx)` for the last temporary LP when the locked LP is another candle (the
  warning case; none on this window); `locked (cycle N+1's BOS confirmed at idx)`, `open (still moving at the data
  end)`, `unlocked (cycle ended by a reversal / sub replaced / sub ended at idx)` unchanged.

## Step 1 — first_counter reads the H1 cycles' first sd proximity trigger (built 2026-10-26)

- **Change:** `multitf/uc1_trigger.detect_uc1_triggers(sorted_events, h1_df, zone_proximity_triggers)` fires
  `first_counter` on every cycle whose FIRST zone-proximity trigger is sd and whose CTS is confirmed, in
  CTS_CONFIRMED order, `trigger_event_idx` = that trigger's candle — from the proximity lists, not the WVMI records
  (it took `wvmi_records` + an unused `kl_zones` before). One reading shared with the main WVMI gate:
  `zones/zone_proximity.first_sd_proximity_triggers` (`orchestrator._first_sd_prox_gate` now calls it).
  `_run_multi_tf_dual` passes `main_zone_proximity_triggers` (the batch lists) and drops its `wvmi_records`
  parameter; the chart's `_zone_render.proximity_sub_trigger_types` reads `attrs["zone_proximity_triggers"]` (the
  final lists — each cycle's first trigger = the batch's, `_final_proximity_triggers` asserts it).
- **Predicted:** no output change (the 3 gated cycles (0,1) 748, (1,1) 820, (1,2) 935 all have a WVMI record, so the
  same 3 first_counter triggers). **Measured:** 27 / 27 CSVs byte-equal to S_repl_break; the three charts' traces,
  shapes and layouts JSON-equal (hovers included); the run log's `[uc1_trigger]` lines the same 3.
- **What differs off this window:** a gated cycle whose WVMI record cannot be computed (no BOS zone, a missing wave
  candle, a zero FB / FP volume — WVMI_SPEC "Edge Cases") now still starts its first_counter sub. Seen in one test
  fixture (`test_proximity_trigger_hover.py`, the H1 POI marker: its cycle's first trigger is sd and the CTS
  confirmed, no WVMI record → `sub_trigger` was `NA (no sub trigger)`, now `first_counter`).
- **Tests:** the full suite 1980 passed; 9 / 9 mutants KILLED (`mutants_wvmi_step1.py`: the helper's first /
  direction reads, the detector's filter / candle / order, the gate's reading, both call sites — the driver's
  `test_run_multi_tf_dual_hands_uc1_the_main_proximity_lists`, the chart's lookup test).

## Round 9 — the user's answer (2026-10-26, verbatim)

> yes hover should say zone_side

Decided: the dot hovers (sub and main) say `zone_side=buy|sell`, not `wvmi_dir=` — one name for the value in the
hover, the per-LP CSV and `_wvmi.csv`. The accepted hover drafts' last line reads
`zone_side=… | breakout_momentum=… | pullback_momentum=…`.
