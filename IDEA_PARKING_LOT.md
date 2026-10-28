# Deferred-Items Register & Idea Parking Lot
Multi-Timeframe Forex Trading System

**The ONE canonical list of everything deferred, parked, "revisit later", or planned-but-unbuilt.** Each entry is one
line: what, its trigger or target week, and a pointer to where the DETAIL lives (a spec section, a plan, a memory file —
the detail is never duplicated here). Ideas are recorded immediately but implemented deliberately; do not implement an
entry outside the syllabus sequence without the user's go.

**How to add / close an entry.** A new deferral: one line here (bucket, trigger, source) + its detail in the owning doc.
Closing one: move its line to "Recently closed" with the date and commit — never delete silently. Memory
(`MEMORY.md` "Next session priorities") names only what is NEXT and points here for the rest.

**Built 2026-09-30** from a full sweep (every repo `.md`, all 51 memory files, code TODO/FIXME comments; grep for
deferral wording, each hit read in context, uncertain ones checked against the code). An item phrased without deferral
wording could still be missing — add it when found. Memory paths are relative to the Claude project memory folder
(`~/.claude/projects/<project>/memory/`).

Status: **OPEN** (to do) · **DECIDE** (needs the user's call first) · **TRIGGER** (waits for a condition) ·
**CONFIRM** (possibly superseded — check with the user).

---

## A. Week 8 close-out — Week 8 DoD SIGNED OFF by the user 2026-09-30 (after bucket C landed); merged to `main`

**Week 8 is done.** The user held Week 8 open until bucket C (Part 4 closure) landed. The DoD itself ("HTF and LTF
context aligns logically; confluence vs non-confluence is visually obvious") was already met. C landed 2026-09-30, and
the same day the user signed off the DoD; `/commit-save` and the merge `week8-volmom-multitf` → `main` followed.
The open lines below carry over (fold in when their files are next touched); D and later come after the merge.

**Chart review with the user** — all items closed 2026-09-30 (see "Recently closed").

**Hygiene** — the 2026-09-30 pass is done (see "Recently closed"); what is left:
- ~~**Charting audit 2026-09-21 — the style / code items**~~ — DONE 2026-10-28 (SESSION ORDER item 8a, PLAN_V §1,
  zero output change): #7 `zone.m15.outline`, #8 `hoverlabel.m15_structure` / `hoverlabel.zone_proximity`, #9 the
  M15 zone hitboxes read `zone.kl.hover_line` / `zone.poi.hover_line` (+ the twins `wave_candle.hover_line`,
  `fib.hover_line` on every chart), #11 the dead key + `watch_y` deleted, the 8 F841 locals (+ their orphans) deleted,
  the unused `zone.h1_overlay.*` + `fib.label` keys deleted. **LONG-TERM PARKED 2026-10-28** (the user at the item-10
  sweep: "any remaining charting items … "long term" parked … unless I specifically mention something about doing
  another detailed sweep" — §D item 10) **— the audit's unverified list, each a rendered-figure change:** the H1 chart draws a collapsed cycle's fib;
  proximity markers on collapsed-cycle zones; the opt-in M15.counter KL overlay on H1 has no collapsed skip; the H1
  chart drops the `wave_candle.*` `dash`; the wave verticals at forming candles take the live style; the H1 chart vs
  H1 overlay filter quirks; a sub whose confirmed points are all forming draws no live segment; PART4 §16.7 (zone text
  labels specified, none rendered — build or drop). Detail: memory `project_charting_audit_20260921.md`.
- ~~**Two dead reads found by the Week 9 /prepare + item-1 talk**~~ — both closed: `wvmi._find_temporary_lp`'s
  `fp_vol_dir` GONE 2026-10-26 with the function ("WVMI back on the chart" step 3, §D item 5);
  `_run_multi_tf_dual(poi_zones=...)` is READ since PLAN_R step 2 (`817d2a9`, `build_cycle_proximity_scans`,
  `pipeline/orchestrator.py` ~:1126) — the entry was stale (found at SESSION ORDER item 8a, 2026-10-28; PLAN_V §1.2).
- ~~**The H1-main frame is classified and stamped "H1" whatever the config timeframe**~~ — DONE 2026-10-28 (SESSION
  ORDER item 8a, PLAN_V §1.3): `run_replay.main()` asserts `CONFIG.timeframe == "H1"` — the orchestrator is H1-main
  throughout (the "H1" classification stamp + floor-seed key, `"H1.main"`, `parent_tf="H1"`), so passing
  `CONFIG.timeframe` to the classifier alone would have made an M15-main run look supported. A re-enabled M15-main
  window (`config.py` keeps them commented out) now fails at the start.

## B. Week 8 scope — decided 2026-09-30 (the M5 layer moved to D; the syllabus extras dropped — "Recently closed")

## C. Part 4 closure — DONE 2026-09-30 (all three items in "Recently closed"; Part 4 is complete)

## D. Week 9 Part 1 — Entry prerequisites (before Entries) — SESSION ORDER items 1–9 DONE 2026-10-28; Part 1 stays OPEN for item 10, the sweep of every parked / deferred item (the user 2026-10-28); D.3 = the items moved to Parts 2 / 3

**Week 9 = "Entries & Decisions"** (renamed by the user 2026-09-30), in three parts: **Part 1** Entry prerequisite
items (this bucket), **Part 2** the actual Entry decision build & logic, **Part 3** Other decisions build & logic
(SYLLABUS Week 9). User: "I agree that these items should be worked on prior to Entries"; a rough outline, details
come from the user as Week 9 reaches each item.

**Order — set by the user 2026-10-01: the heavy items first (D.1), the rest after (D.2).** Verbatim: "I would like
to tackle the heavy lifting items first. These will be ones with extensive documentation, build and changes to
structure elements that could differ a lot from our current iteration. Each of these may require separate design /
planning and build sessions." Items 1 and 3: the user provides updated detailed specs. Items 2 and 4: "may involve
flipping across multiple different scenarios & parameters and reviewing the output/charts to get to the final output
we want." "The remaining items you listed we'll keep in the backlog and decide on them after the above items are
complete." Item 1 opens the next working session. **Re-ordered by the user 2026-10-06:** item 1's chart debugging
PAUSED; item 2 — now the cross-cycle fib revamp — NEXT; item 1's debugging resumes after it (PLAN_H §6 F2a–F2b).

**SESSION ORDER — set by the user 2026-10-12 (after PLAN_N landed; supersedes "item 3 next").** Verbatim: "For the
very next session, I want to land the H1 range-hover follow-up first. Then the session after that, I want to go back
to the current IC identification and cross cycle fib logic to ask some questions and make some edits. After that, the
next session will focus on some changes & edits on zone proximity triggers, more specifically the current opposite
direction / CTS zone proximity trigger. Finally, after that we will start zone size adjustments."
**UPDATED by the user 2026-10-20 (after PLAN_R landed) — item 3 DONE, a new item 4 before zone size adjustments.**
Verbatim: "For now let's consider item 3 done. We could potentially revist down the line when I notice something else in the future. Before moving on to zone adjustments, I'd like to go back and revisit some things around IC identification logic. I will provide further specs for the changes in future sessions."
**UPDATED by the user 2026-10-21 (after PLAN_S landed) — item 4 DONE (closed; may be revisited), a new item 5 before
zone size adjustments.** Verbatim: "I was simply saying that we may come back to IC if in the future I see something wrong or want to fix. BUt for now, we can consider it closed. Just don't be surprised in the future if we come back to it again." And: "for the very next session, I would like to actually work on a few charge updates & edits before we move on to zone adjustments." ("charge" = chart.)
**UPDATED by the user 2026-10-27 (after "WVMI back on the chart" landed) — the cycle-0 carve-out revisit NEXT, then item
5's final part.** Verbatim: "I will give you a bit of a preview of what I'd like to work on in the next few sessions.
First, I want to revisit the cycle-0 carve-out that we discussed recently. Afterward that, we will complete the final parts of charting updates, which will consist of 1) adding a toggle to hide parent (1H) chart overlay elements, and 2) adding an ability to zoom into certain parts of the chart via a) a +/- switch or tool and b) allowing user to input an idx for the chart to zoom into. We can discuss in details those things when we get to them." (Zone size adjustments, item 6, not restated — they stay after item 5.)
**UPDATED by the user 2026-10-27 (after item 5 completed) — item 6, zone size adjustments, MOVED out of Part 1 into
Parts 2 / 3 (D.3 below); the pinbar body-pip floor calibration NEXT (item 7), the last major Part 1 item; then the
remaining minor pre-Part-2 items (item 8); the entry confirmation candle patterns MOVED to Part 2 (D.3).** Verbatim:
"with that we've now completed all the charting updates I wanted to make for now. At this point we are close to being
done with part 1 of week 9." "Now originally I wanted to work on zone adjustments as part of Week 9 Part 1, however
I've since changed my mind. We will incorporate this directly into entry & decisions (Part 2 & 3) rather than as a
separate item that adjusts existing zones." "Instead, the last major item I want to tackle in the next session is the
Pinbar body-pip floor calibration. Afterwards, we can take care of any remaining minor tasks & items in the next
session(s) that we previously documented as pre-reqs to starting part 2 of week 9. We will also move entry candle
patterns to part 2 of Week 9, thus allowing us to complete part 1." The rationale for moving zone sizing (verbatim):
SYLLABUS Week 9 Part 2 "Zone size at entry and stop-loss".
1. ~~**The H1 range-hover follow-up**~~ — **LANDED 2026-10-12** (session 1; `9f9406d`, S_N_hover): widened by
   the user to option B (the hover line = the box, candle for candle) + the dead passes deleted; `/compare` vs S_N = the
   prediction (4 H1 hover traces, 28 candle points). Register §A; `engine_v2/plans/plan_n_inputs/
   followup_hover_per_structure.md` "The landing decision".
2. **The IC identification + the cross-cycle fib logic — the user's questions and edits** (re-opens D.1 items 1 / 2 —
   both had no open work since 2026-10-12; the user brings the questions). Current logic, canonical:
   `engine_v2/zones/POI_ZONES_SPEC.md` §3 (Step 1 the IC conditions, §3.2 the IC reference, Step 2 the institutional
   pattern + its tiebreak rules, Step 3 the ranking rules; Step 4 deferred), `engine_v2/zones/CROSS_CYCLE_FIB_SPEC.md`
   (§3 the cross-fib limit incl. a counter sub's `parent_bos_threshold`, §8 MS's read); plans PLAN_H / K / L, PLAN_I / J /
   M. Each edit = its own plan / `/compare`. **DONE (sessions 2–3, 2026-10-12 → 13):** the user's three-change
   write-up (`engine_v2/plans/qa_ic_crossfib_inputs/user_writeup_20261013.md`; Q&A + measurements `qa_findings.md`) —
   the user's split: **change 2 LANDED 2026-10-13 = PLAN_O** (`engine_v2/plans/PLAN_O_ic_reference_target_cycle.md`: a
   cross's IC CTS reference = its TARGET's previous CTS; source `ce5ac9b`, save S_O `20261013_151425_ce5ac9b`);
   **changes 1 + 3 LANDED 2026-10-13 = PLAN_P, ONE commit** (the revert unit, the user 2026-10-13;
   `engine_v2/plans/PLAN_P_ic_condition3_step1a.md`): IC condition 3 = the immediate unfilled imbalance (≤ IC + 2;
   the immediate rule left the Step 2 / Step 3 rule sets) + Step 1A, the 80–100 fallback zone when Step 1 finds no
   candidate (its POI's activation re-checks 80–100); the POI activation's imbalance condition stays as before,
   decoupled from IC identification; the old condition 3 kept as information; the BOS anchor candle may be a Step 1A
   IC (the user: "Keep it"). `/compare` vs S_O = the prediction byte-for-byte (H1 POIs 5 → 2, one H1 cascade 939 →
   935, 4 Step 1A POIs, 0 M15 structure-event changes); mutants 27 / 27; the user's chart look: "Charts look good".
   Source `23f395d`, save S_P `20261013_231327_23f395d` = `LATEST_<branch>` (the `/compare` skill's PLAN_P entry).
   Decisions verbatim: `qa_findings.md`. **NEXT = SESSION ORDER item 3** (the opposite-direction trigger).
3. ~~**Zone proximity triggers — changes and edits to the opposite-direction (CTS-zone) trigger**~~ — **DONE
   2026-10-20** (PLAN_Q + PLAN_R; the user: "consider item 3 done … could potentially revist down the line"). (the user brings the
   details). Today: `engine_v2/zones/zone_proximity.py` — per (sid, cycle) with a `CTS_CONFIRMED`, triggers alternate
   sd (BOS / POI zone) → opp_sd (the CTS zone's inner bound) → sd …, at H1 8 / M15 5 / M5 3 pips; narrow cycles: Rules
   1–3 (LANDMINES "Narrow-Cycle Rules 1+2+3 Are a Triple"). The opp_sd (opp_sd-prox — "CTS-prox" before PLAN_R — "var 3") entry is what
   `multitf/subsequent_confluence_trigger.py` fires on (one `subsequent_confluence` per opp_sd entry preceded by an sd),
   and `multitf/subsequent_counter_trigger.py` requires it inside the [sd, opp_sd, sd] window of a `subsequent_counter`
   (var 4) — so an opp_sd change reaches BOTH lenses' subs. Re-read the code + PART4 §4.3.4 / §4.3.5 at the session
   start. **First landing 2026-10-14 = PLAN_Q, on the sd side** (`engine_v2/plans/PLAN_Q_proximity_poi_like_kl.md`;
   the user: "For the purpose of proximity triggers, I would like POI to be treated exactly like a KL zone"): the scan
   reads a POI by LOCATION at every candle of the scan, never its activation history. Before it, a collapsed cycle's
   POI never counted (its history is floored at the structure's lifecycle start, so empty) while its equally inactive
   BOS zone did: S_O's sid 1 cyc 1 fired 820 sd:BOS beside its POI (IC 732, the nearer inner) — sd:POI under the new
   rule; (0,1)'s POI counted. Two asserts pin the bounds (the floor ≤ the scan start, the end past the scan).
   `/compare` vs S_P = 0 / 27 CSVs (PLAN_P had already removed that POI), 3 hover cells (the (1,2) 1035 sd:POI marker's
   `zone_confirmed_idx` 998 → 902 — now the POI's unclamped activation floor, not its active stretch's start);
   mutants 14 / 14; the user's chart look: "Charts look good". Source `da8c15e`, save S_Q `20261014_170919_da8c15e` =
   `LATEST_<branch>` (the `/compare` skill's PLAN_Q entry). **Second part, UNDER WAY 2026-10-19 = PLAN_R** (the
   user's opposite-direction changes: proximity reference ladders with sub zones, passed proximity zones, the narrow
   proximity-reference cap 40 / 25 / 15, the proximity overlay inside the lifecycle sweep; plan
   `engine_v2/plans/PLAN_R_opp_sd_zones_proximity.md`, Q&A rounds 1–9 `engine_v2/plans/qa_opp_sd_zones_inputs/`):
   READY TO BUILD 2026-10-20 — every decision closed (Q&A rounds 1–12), the plan review done (3 agents, 0.84M) and
   folded in; build order 0a → 0b → 1 → 2, 4 commits, `/commit-save` only after step 2. **BUILT (was UNDER WAY)
   2026-10-20** (PLAN_R §8 = the build log): steps 0a `a6e71c0`, 0b `c1bd05a` and 1 `d26208e` (the ladders + the narrow proximity-reference cap on parent zones) LANDED, all zero change as predicted (the landing-review follow-up `77f8ccf`: 0a's bound = the parent cycle's CTS-established moment); step 2 (the overlay inside the sweep) LANDED 2026-10-20 `817d2a9`, save S_R2 `20261020_115859_817d2a9` = `LATEST_<branch>` (the `/compare` skill's PLAN_R step 2 entry) = the §5.4 prediction (H1 triggers 6 → 10, subs 8 → 11) + its landing review (the source switch: a contributed + reachable zone); **PLAN_R step 3 = the user's round-15 pivot — LANDED 2026-10-20 `d424099`, save S_R3 `20261020_225749_d424099` = `LATEST_<branch>` (PLAN_R §9): the proximity reference gate — a reference update is made only if the gap after it stays above 40 / 25 / 15 (no trigger counters); the only cap left, the narrow parent-reference exception (a parent BOS–CTS inner gap ≤ the minimum); an sd sub zone is no candidate for the sd reference before the cycle's first sd trigger (round 16). PLAN_R COMPLETE (steps 0a–3); the user then closed item 3 (2026-10-20) — next in the SESSION ORDER: item 4, the IC identification revisit (the user's spec to come), then item 5, zone size adjustments**.
4. ~~**Revisit the IC identification logic**~~ — **DONE 2026-10-21** (PLAN_S; the user: "for now, we can consider
   it closed. Just don't be surprised in the future if we come back to it again") (the user, 2026-10-20: "go back and revisit some things around IC
   identification logic. I will provide further specs for the changes in future sessions"). **Spec received 2026-10-20
   (`engine_v2/plans/qa_ic_revisit_inputs/user_writeup_20261020.md`), decided through Q&A round 5 2026-10-21
   (`qa_findings.md` there, rounds 1–7): `engine_v2/plans/PLAN_S_fill_rule_and_ic_step1b.md` COMPLETE 2026-10-21
   (steps 1–3 LANDED, each = its prediction exactly: `69c099b` save S_S1, `cbaed31` save S_S2, `7b62a47` save S_S3
   `20261021_130400_7b62a47` = `LATEST_<branch>` — the `/compare` skill's PLAN_S entries; item 4 closed by the user
   2026-10-21, may be revisited) — step 1 the
   imbalance arming threshold 70 % → 65 %, step 2 the two fill strokes on separate candles (both engine-wide), step 3
   IC Step 1B (prefer candidates whose closest prior imbalance is unfilled through the candle before the IC); IC
   condition 1 unchanged.** Current logic, canonical: `engine_v2/zones/POI_ZONES_SPEC.md` §3 (Step 1 the IC
   conditions incl. PLAN_P's condition 3 = the immediate unfilled imbalance and Step 1A the 80–100 fallback zone,
   §3.2 the IC reference incl. PLAN_O's target-cycle CTS for a cross, Step 2 the institutional pattern + its
   tiebreak rules, Step 3 the ranking rules; Step 4 the lower-TF fallback DEFERRED — D.2 "Step 4 of IC
   identification"); the decisions behind it: PLAN_H / K / L / O / P and `engine_v2/plans/qa_ic_crossfib_inputs/`.
   Each change = its own plan, prediction and `/compare` (as PLAN_O / PLAN_P).
5. ~~**Chart updates & edits**~~ — **DONE 2026-10-27** (its edits + the final part, the control bar — below). Was
   NEXT, the very next session (the user, 2026-10-21; the user brings the list). Chart
   items already on file — background only, none in scope unless the user names it: §G "Overall chart revamp &
   improvements" ((1) the fib hover masks candle hovers, (2) the proximity markers' `zone_confirmed_idx` means two
   things), §G "Uniform 3-tier chart opacity", §G the stepped "current proximity reference" line on the H1 chart
   ("Decide later"), §A the charting-audit style items #7 / #8 / #9 / #11 + 8 unused locals in
   `charting/export_m15_chart.py`. Canonical rules: `engine_v2/charting/CHARTING_SPEC.md`, all styling in
   `style_registry.py`. **IN PROGRESS (2026-10-22):** the first edit LANDED — the BOS / CTS dot hover, main + sub (the
   user's spec + decisions + what landed: `engine_v2/plans/qa_chart_edits_inputs/user_spec_20261022_dot_hover.md`;
   figure-only; source `ab2fcc1`, save S_dot_hover `20261022_200747_ab2fcc1`). Its DEFERRED quirks (PB `cycle_id=0`, PB = pullback or replacement break, the confirmation moment /
   method not shown, …): that file's "Quirks" section (the user: "Let's come back to it later to address them in more
   detail"). The second edit LANDED 2026-10-23 — the KL / POI zone hover, main + sub + the H1 overlay (the user's
   spec + decisions + what landed: `engine_v2/plans/qa_chart_edits_inputs/user_spec_20261023_zone_hover.md`;
   figure-only; `lifecycle=` = the zone's own active window, every IC line labelled; source `56bb20d`, save
   S_zone_hover `20261023_153743_56bb20d`). The third edit LANDED 2026-10-23 — the Fib zone hover, main + sub (the
   user's spec + decisions + what landed: `engine_v2/plans/qa_chart_edits_inputs/user_spec_20261023_fib_hover.md`;
   figure-only; `ic_ref:` / `limit_price=` / `lifecycle=` relabelled, the IC outcome line kept only on a fib with no
   IC; source `563f5f8`, save S_fib_hover `20261023_191547_563f5f8`). The fourth edit LANDED 2026-10-23 — the wave
   candle hover, H1 + M15 sub + the H1 overlay (the user's spec + decisions + what landed:
   `engine_v2/plans/qa_chart_edits_inputs/user_spec_20261023_wave_candle_hover.md`; figure-only; the window +
   `struct_direction` added, `volume=`; `debug/chart_census.py` fixed — the hover edits had broken it; source `18711dc`, save S_wave_hover
   `20261023_194643_18711dc`). The fifth edit LANDED 2026-10-24 — the zone proximity trigger marker hover, H1 + the
   M15 charts' H1 overlay (the user's spec + decisions + what landed:
   `engine_v2/plans/qa_chart_edits_inputs/user_spec_20261023_proximity_marker_hover.md`; figure-only;
   `trigger_price` / `zone_inner` / `sub_trigger` added, `zone_kind` removed, the M15 copy keyed by its parent; source
   `fb0189b`, save S_prox_hover `20261024_105742_fb0189b`). The
   M15 Prev BOS line hover stays as is (the user, 2026-10-24: "no changes needed for Prev BOS line hover"); WVMI back
   on the chart TABLED (register §G). **The hover-label pass CLOSED 2026-10-24** (the user: "We've focused mostly on
   marker & data hover labels in this session. Next session I would like to continue to debug the charts but now I
   will focus on other elements"); its conventions and the hovers it did not touch (background only, not in scope
   unless named): CHARTING_SPEC "Hover label conventions". NEXT: debugging the charts' other elements (the user brings
   the list). **Second pass, the charts' other elements (2026-10-24 →):** the first edit — a replaced sub's line breaks
   at the same point on every lens (the user: confluence sub_3 "go from 3047 to 3304 then down to 3490"; the cause:
   the replacement break read the replacer from THIS lens's records, so a lens whose record ended after the sub drew
   none; the user's decision A, one replacer per sub, pool-wide — also counter sub 5 → 3621 → 3760 → 3819; the
   user's request + decision + what landed: `engine_v2/plans/qa_chart_edits_inputs/user_spec_20261024_replacement_break.md`;
   figure-only, +1 PB dot per M15 chart; canonical CHARTING_SPEC "Replaced subs run through one more point"; source
   `56517e7`, save S_repl_break `20261024_171443_56517e7`; the user: "charts look good go ahead"). The second edit —
   **WVMI back on the chart** (UN-TABLED by the user 2026-10-24; CLOSED 2026-10-26 — "Recently closed"): the design settled in nine rounds 2026-10-24 →
   26 (the user's request, every answer verbatim, the measurements and the decisions:
   `engine_v2/plans/qa_chart_edits_inputs/user_spec_20261024_wvmi_on_chart.md`) — dots above the volume bars at each
   temporary LP, a furthest-close LP rule with first-LP rules by how the CTS confirmed, no H1 gate, a per-LP CSV per
   chart. Five landings in the user's order, one cause each: **1** `first_counter` reads the H1 cycles' first sd
   proximity trigger directly, not the WVMI records — LANDED 2026-10-26 (zero change; source `257f8ba`, save
   S_uc1_prox `20261026_175906_257f8ba`); **2** the H1 WVMI without its
   gate (+2 H1 records, (0,0) and the collapsed (1,0); the main's `triggered_by_*` now attribution, None without a
   first sd trigger; the mode name `"first_sd_prox"` kept by the user) — LANDED 2026-10-26, = the prediction (source
   `76b5a17`, save S_h1_ungated `20261026_183615_76b5a17`); **3** the new LP rule (the furthest-close sequence with
   its first-LP routes, never FP, to the cycle's end − 1; `pattern_anchor_idx` on a pullback CTS_CONFIRMED; a
   `[wvmi] WARNING` when the locked LP differs from the last temporary one — kept by the user) — LANDED 2026-10-26, =
   the prediction (6 records' final LP + pullback momentum move, the structure-events CSVs gain the key — a
   save-format boundary; the warning prints 10 times for one collapsed record, predicted 2; source `f63df34`, save
   S_lp_furthest `20261026_192655_f63df34` = `LATEST_<branch>`); **4** the per-LP rows +
   `*_wvmi_lps.csv` — LANDED 2026-10-26, = the prediction (one row per dot: H1 57 / counter 74 / confluence 185;
   collapsed records get none — the user, "I may change my mind later", §G; an LP before its sub's `start_idx` keeps
   its row with `relative_dir = NA (before start_idx)` — the user, round 11; 27 → 30 CSVs, the 27 byte-equal, charts
   unchanged; source `588014a`, save S_lp_rows `20261026_220225_588014a`); **5** the dots + hovers on the three
   charts — LANDED 2026-10-26 (source `2dc9e5d`, save S_wvmi_marks `20261026_230619_2dc9e5d` = `LATEST_<branch>`), =
   the prediction (figure-only: 30 / 30 CSVs byte-equal; traces added only — H1 +2,
   each M15 chart +4; one mark per per-LP row above the volume bars, sell red just above the bars / buy green one step
   higher, a sub's a dot, an H1 LP a circle — 57 per chart, on the M15 charts at the hour's last M15 candle; the
   hover = the accepted drafts, the H1 circle's M15 copy `TF=1H>>15M` — the user, round 12; the volume axis 1.4 × the
   bars' max; canonical CHARTING_SPEC §13); the user's look: "the positioning looks good", the circles "a bit
   smaller" (11 → 9 → 7 → 8, "let's go with 8 as the final value"). **The second edit, "WVMI back on the chart",
   is COMPLETE.** Paused 2026-10-27 for the cycle-0 carve-out revisit (5b below — DONE 2026-10-27). **Item 5's FINAL
   part — DONE 2026-10-27, the chart control bar** (the user's ask: "1) adding a toggle to hide parent (1H) chart
   overlay elements, and 2) adding an ability to zoom into certain parts of the chart via a) a +/- switch or tool and
   b) allowing user to input an idx for the chart to zoom into"): HTML controls above each chart's title (the PNGs
   unchanged) — a **Parent (1H)** button on the M15 charts hiding / showing everything `_render_h1_overlay` draws (60
   traces + 40 shapes per M15 chart, `legendgroup = "parent_overlay"`; NOT the H1 trigger markers or the H1 WVMI
   circles — the user, "All except both"; PART4 §16.4's legend group built as this button instead); time-only **− / +**
   and the wheel, with a price re-fit; a **Box zoom | Pan** switch (pan keeps the window's candle count across
   rangebreaks); an **idx box** (+ a 15M / 1H selector on the M15 charts) centring 48 candles either side with a blue
   highlight that stays until the next jump (Go on an empty box clears it — the user, round 2). = the prediction
   (figure-only: 30 / 30 CSVs byte-equal, the H1 figure equal, each M15 chart's 60 traces + 40 shapes differ only by
   that `legendgroup`, + the control script); the user's look: "charts look good, go ahead with the commit flow".
   Source `5dfb5b3`, save S_controls `20261027_164527_5dfb5b3` (= `LATEST_<branch>` until S_U1, item 7; save `32cde54`, trunk `6ab4b62`;
   the `/compare` skill's entry).
   Canonical: CHARTING_SPEC §14; every round verbatim + the measurements:
   `engine_v2/plans/qa_chart_edits_inputs/user_spec_20261027_parent_toggle_and_zoom.md`. **Item 5 is COMPLETE — NEXT:
   item 6.**
5b. ~~**Revisit the cycle-0 carve-out (no sd-proximity CTS confirmation on any cycle 0)**~~ — **DONE 2026-10-27**
   (PLAN_T, `engine_v2/plans/PLAN_T_probe_parity_and_cycle0_carveout.md`): the carve-out stays on the main sid 0 and
   on the sub the first_confluence of the main sid 0 cycle 0 creates (+ that trigger's probe), dropped on every other
   cycle 0; the first_confluence probe's MS reads what its sub reads (M15 5 / 30, pip size, the sub's cross-fib
   limit). T1 the probe — LANDED, zero change; T2 the carve-out rule — LANDED = its measured prediction (subs 4, 7, 8:
   cycle-0 CTS confirmed by proximity at 3336 / 4071 / 4098; the user's look: "Looks good, commit T2"; no
   confirmation mark on the M15 charts: "Not needed"). Canonical rule: MARKET_STRUCTURE_SPEC "Cycle-0 carve-out" +
   LANDMINES "Dual CTS Proximity Confirmation — the Cycle-0 Carve-out (main sid 0 + its FC sub) + Rule 1". Every
   round verbatim + the measurements: `engine_v2/plans/qa_cycle0_carveout_inputs/user_spec_20261027_cycle0_carveout.md`.
   Was **NEXT** (inserted by the user
   2026-10-27 before item 5's final part; moved here from §G, where it was parked 2026-10-26). **DECIDE**; raised by the
   user 2026-10-26 with "WVMI back on the chart": "yes we do need to add line to revisit this. I need to better
   understand the original carve out logic. A decision from May was also quite a long time ago and a lot of things
   have changed since then … We can come back to it in more detail later". The rule: MARKET_STRUCTURE_SPEC "Cycle-0
   carve-out (proximity disabled)" + LANDMINES "Dual CTS Proximity Confirmation — Cycle-0 + Rule 1 Are BOTH Required"
   (the titles then; both gates kept by the user's 2026-05-12 decision). The user's concerns: it reads as built for the main sid 0
   cycle 0 (other cycle 0s come from the engine's own structure logic and triggers); single-cycle unique subs must
   all confirm by a pullback pattern. Case: confluence sub 7 cycle 0 (a 57-pip gap) sat at its POI from 4071 and
   confirmed only at 4157. Detail + facts: `engine_v2/plans/qa_chart_edits_inputs/user_spec_20261024_wvmi_on_chart.md`
   rounds 6–7 ("Measured for round 7 — facts for the carve-out revisit").
6. ~~**Zone size adjustments**~~ — **MOVED 2026-10-27 to Parts 2 / 3** (the user, after item 5: "We will incorporate
   this directly into entry & decisions (Part 2 & 3) rather than as a separate item that adjusts existing zones") —
   D.3 below; the zones keep their base form. Was D.1 item 3.
7. **Pinbar body-pip floor calibration** — **DONE 2026-10-28: PLAN_U** (the user 2026-10-27: "the last major item I want
   to tackle in the next session"; 2026-10-28 the approach chosen — a rolling floor, the p90 of the 90 most recent
   pinbars; U1 the per-candle plumbing + columns + hover LANDED 2026-10-28, format only; U2 the rolling floor, natural
   window, LANDED 2026-10-28 = its prediction (S_U2 `20261028_113834_84fbbc6`); U3 the final window — the user's pick
   after seeing both ("Final + convergence check"; natural kept as a switch) — LANDED 2026-10-28 = its prediction
   (S_U3 `20261028_123113_32c544d`); U3b, the floor seed guard steps back until final mode converges — the user:
   "Guard extends" — LANDED 2026-10-28 = its prediction (only the two `[floor_seed]` lines change, every output
   byte-equal; S_U3b `20261028_130713_9c3949d`, source `9c3949d`; PLAN_U §6b / §9 "U3b")) — D.1 item 4 below.
   NEXT: item 8.
8. **The remaining minor items before Part 2** — after item 7, in the next session(s) (the user 2026-10-27: "any
   remaining minor tasks & items in the next session(s) that we previously documented as pre-reqs to starting part 2
   of week 9"). **SCOPE DECIDED by the user 2026-10-28** (after item 7 closed; shown all 13 candidates of §A + D.2
   with a size / expected change / recommendation each, the user picked — AskUserQuestion — every recommended
   option): **(8a)** the §A hygiene trio — the charting audit's style / code items (#7, #8, #9, #11 + the 8 unused
   locals in `export_m15_chart.py`), the dead `_run_multi_tf_dual(poi_zones=…)` read, the H1 frame's "H1" stamp
   (assert / pass `CONFIG.timeframe`) — expected zero output change; **(8b)** the same-candle replacement chain's
   `ended_by` = the replacer that survives (D.2) — 0 on the window, expected zero change; **(8c)** the agreed direction
   for wave candles + WVMI (D.2, PLAN_H Q-W) — a behaviour change: its own plan, prediction, `/compare`, chart look.
   Order (proposed by Claude — the zero-change landings first, so the behaviour change's `/compare` carries one
   cause; the user may reorder): 8a → 8b → 8c. Not picked, stay in D.2: the missing-label fill paths, B5, the
   "sub 3 cyc 3 has no POI record" note (item 1's debugging log; re-measured 2026-10-28 on S_V8c: its case — an
   ended, never-locked sub fib, which the POI record gate skips — has 0 instances on either lens; the 4 locked sub
   fibs without a POI row are `no_ic_candidate` / `all_ic_candidates_pinbar` outcomes —
   `engine_v2/plans/plan_v_inputs/backlog_no_poi_census.py` / `.txt`), the sub probe reference = the
   derived CTS KL zone. IC Step 4: KEEP DEFERRED. Moved to D.3 (Parts 2 / 3): the candle-direction / rejection
   confirmation, zone strength scoring, the M5 layer. Then Part 1 was to be complete — the user added item 9 after
   item 8 closed (below).
   **Progress 2026-10-28 (PLAN_V, `engine_v2/plans/PLAN_V_part1_minor_items.md`):** 8a DONE (`955489b`, zero change)
   and 8b DONE (`01f5282`, zero change) — the user: "Two commits, one save" (S_V8b `20261028_141322_245620c` = S_U3b
   byte for byte); 8c MEASURED (PLAN_V §3.1: 0 of 53 wave
   candles move, sub 3's temporary-LP history gains three volume-only dojis) and DECIDED: "Apply it" + a
   direction-0 wave candle drawn in its `vol_dir` colour ("Draw it in its vol_dir colour"); 8c LANDED `1318c82` (= the
   variant byte for byte, PLAN_V §3.3; the user's chart look: "looks good go ahead"), save S_V8c
   `20261028_153518_1318c82`. **Item 8 DONE.** NEXT: item 9.
9. **The last Part 1 items — three of item 8's not-picked backlog entries** (the user 2026-10-28, after item 8 closed,
   shown a brief description of each of the four: "I think once we complete these we will truly be done with W9 Part
   1"; the user's words per entry, verbatim):
   **(9a)** the missing-label fill paths (D.2 below) — "Let's address in next session". Zero output change expected
   (unreachable: every engine frame is a contiguous index).
   **(9b)** the POI record gate for a fib that ENDED without ever LOCKING (was the "sub 3 cyc 3 has no POI record"
   note) — "This one I think I'd like to also address, specifically the part around A sub fib that ended without ever
   locking gets no POI record, so no IC is ever evaluated for it. I'm not sure when we implemented this specific rule,
   but I'm not sure that is what I want if you are saying anytime fibs are not locked we don't get an IC or POI."
   The rule as coded (`zones/poi_zones._poi_record_gate` + the next check in `select_ic_records`, main and
   subs alike; FIB_LIFECYCLE_SPEC §11): a fib record gets a POI record if it is LOCKED (a fib locks at its cycle's
   CTS_CONFIRMED) OR still LIVE in the final records (condition-active AND its cycle not ended) — AND its cycle has a
   CTS_ESTABLISHED (the user's 2026-09-29 zones-audit option N). So an unlocked fib is not always without one: it has
   it while its cycle is live; a cycle that ENDS before its CTS confirms loses it — and the export / chart are the
   final records, so that cycle shows no POI and no IC at all, not even for the candles it was live. Origin: Week 7
   (`0033c53`, 2026-01-31: "A Fib is valid for IC detection if it's active OR locked" — `active` then went False when
   a cycle ended); FibState Session 2 (`310c395`, 2026-05-27) kept the effect when `active` became condition-only
   (`end_idx is None` added). On this window: 0 instances (`plan_v_inputs/backlog_no_poi_census.txt`; the 2026-10-06
   example, sub 3 cyc 3 3304 → 3611 ended `parent_end`, is gone) — a rule change needs a synthetic case or another
   window for its chart look. The user's call: what a never-locked ended fib should get (a POI with its lifecycle
   ending at the cycle end? an IC evaluated but no POI? the CTS_ESTABLISHED requirement kept?).
   **(9c)** the sub probe reference = the derived CTS KL zone (D.2 below) — "On this I'd also like to discuss more
   next session. Speicifically, for subs are we currently not deriving any CTS zones? So this isn't just a case of
   the zones are derived and simply not being charted?" The answer, checked 2026-10-28 (code + S_V8c): subs DO derive
   CTS KL zones — `orchestrator._run_downstream_pipeline` derives the full set (`all_kl_zones`, BOS + CTS) because
   the CTS wave candles and WVMI need it — then RETURNS only the BOS ones (`source_kinds=("BOS",)`,
   `pooled_structure_build`), so the CTS zones are used internally but are not exported (the sub KL CSVs: confluence
   21 BOS / 0 CTS, counter 8 / 0; H1: 5 BOS / 5 CTS), not charted, and not handed to the probe — whose pool-path
   readers pass `kl_zones=[]` anyway (Plan C, GOTCHAS "the `kl_zones=[]` pool-path note"), so every sub reference
   zone is built ad hoc from the CTS event. BOS-only for subs since the multi-TF foundation (`6a53948`, 2026-02-17;
   ARCHITECTURE "KL zones are BOS-only for subs"). Two separable questions: chart / export the sub CTS zones? feed the
   derived CTS zone to the probe (a behaviour change — sub start candles may move)?
   **Parked by the user (stays in D.2):** B5 — "I think this case is not possible, so let's park for now".
   Order (proposed): 9a (zero change) → 9b (the user's decision, then a plan) → 9c (discussion first). Then Week 9
   Part 1 is complete → push (the user: "don't push just yet") → Part 2.
   **Progress 2026-10-28 (PLAN_W, `engine_v2/plans/PLAN_W_part1_last_items.md`):** 9a DECIDED ("Require 0..n−1" —
   both fill checks raise on a frame not indexed 0..n−1, the label-based code deleted) and BUILT = its prediction
   (zero change vs S_V8c, PLAN_W §1.5), committed `35e827e` (the user: "Commit now, save after 9b"). 9b MEASURED
   (PLAN_W §2.2: 0 on the reference window; 0–2 sub cycles per other window, never the H1 main; the proximity views
   unaffected — the overlay reads a sub's POI only as of its CTS_CONFIRMED) and DECIDED: "Full POI record" (the gate
   `active OR locked` — a cycle that ends before its fib locks gets its IC rows + its POI, ending at the cycle's
   end) + the CTS_ESTABLISHED requirement "Keep it" (PLAN_W §2.3); BUILT `aff21f8` = its prediction (zero change
   on the reference window; 2025-08-01 → 09-20: sub 2 cycle 1's POI, PLAN_W §2.5; the user's chart look: "charts
   look good, go ahead and commit-save"). One save after 9a + 9b: S_W9b `20261028_165600_aff21f8` (= S_V8c byte for
   byte). **9a + 9b DONE.** 9c MEASURED (PLAN_W §3.2) and DECIDED: "Export + chart, toggle off" (9c-1) and
   "Derived, as of the read" (9c-2); 9c-2 BUILT = its prediction (zero change — the one sub reference that differs,
   +2.1 pips outer at 3819, moves no start; PLAN_W §3.5); 9c-1 BUILT = its prediction (the lens KL CSVs + their CTS
   rows, the M15 charts + the hidden `sub_cts_zones` group and its `Sub CTS zones` button; PLAN_W §3.7; the user's
   look: "thank you this look good. please proceed"); one save after both: S_W9c `20261028_180228_538cd2f`. After
   it, the user's KL-hover follow-up: every KL zone hover (main + subs) shows `kind=BOS` / `kind=CTS` (PLAN_W §4
   `7c9477e`, S_W9d `20261028_182318_7c9477e`). **Item 9 DONE.** Part 1 NOT closed yet — item 10 (below).
10. **The parked / deferred items revisit — a sweep, then the user's pick** (the user 2026-10-28, after item 9, at
   the session's close, verbatim): "Before ending this session and also declaring W9 part 1 closed, for the next
   session I actually want to make another revisit on the parked items. In the next session, I want to start with a
   sweep of all parked and deferred items we've noted across all our documentation. I want them listed out with a
   short description of what they are. Please exclude any items that we've already agreed would be covered in Part
   2 & 3 of Week 9. After you provide the list, I will decide if there are anything else I want to cover before
   moving forward to Part 2 of Week 9. Part 2 is an important delineation from Part 1, so I want to make sure our
   overall MS is totally solid before moving on to Part 2." So: (1) SWEEP every parked / deferred / open /
   "revisit later" item — this register's buckets A, D.1, D.2, E, F, G, H, I (+ the open lines of A) as the spine,
   cross-checked against every repo `.md` (specs, GOTCHAS, LANDMINES, plans' "deferred / parked / follow-up / not
   folded" notes), the memory files and code TODO / FIXME — anything found that is not in this register gets its
   line here (the register is the ONE list); (2) EXCLUDE what is agreed for Week 9 Parts 2 / 3 (D.3 + SYLLABUS
   Week 9 Parts 2 / 3); (3) LIST each remaining item with a short description (what it is, where it was parked and
   why, its trigger / target, its detail pointer) grouped by area, marking the ones that touch market-structure /
   zone correctness (the user: "make sure our overall MS is totally solid") and the ones scheduled for Week 10 —
   saved with its script into the plan's inputs folder; (4) the user picks what to cover before Part 2. Part 1
   closes after that. The push stays the user's call ("no need to push yet", 2026-10-28).
   **Progress 2026-10-28 (the next session):** the SWEEP DONE — `engine_v2/plans/part1_sweep_inputs/part1_sweep.md`
   (the full list, 7 groups, ★ = MS / zone correctness; the grep `sweep_grep.py` → `sweep_grep_output.txt`, 1,085
   hits in 121 files, each read in context; the on-window facts `sweep_checks.py` → `sweep_checks_output.txt`). 10
   items were not in this register and got their lines (D.2: L5b, the `REVERSAL_CANDIDATE` `pattern` "?"; E: the
   full-frame imbalances residual, the probe iteration cap, the provisional pinbar floor; G: the dot-hover quirks, the
   hovers not yet on the conventions, M15 elements on H1, Plan E's optional renames; I: PART4 §15's validation
   strategy); 5 stale notes fixed (the file's last section). **The user's pick (2026-10-28, AskUserQuestion):**
   (1) "Measure 3 MS risks" — the probe's BOS_0 read past its bound, MS's POI-inner snapshot vs as-of, fibs dropped
   when a gap completes a candle late: measured on the reference window + other windows, no code change, fixes decided
   after; (2) "Fix the '?' pattern meta" — its own `/compare`; (3) "Re-check parked latent cases" — the BOS on a
   reversal's apply candle, the double RANGE_STARTED, the cap-straddling reversal candidate, the full-frame imbalance
   residual, counted on other windows; not picked: the POI-less re-count. PART4 §15: "Close it" (closed above).
   **Charts — LONG-TERM PARKED** (the user, verbatim): "The overall charts look good for now. We can consider any
   remaining charting items mentioned in the documentation to be "long term" parked. These we may not come back to
   unless I specifically mention something about doing another detailed sweep." — every chart / hover item in §A
   (the charting audit's unverified list) and §G; do not raise them again unasked.

**D.1 — Ordered (the heavy items):**
1. **One POI per cycle** (redefine the IC & POI zone logic) — B landed; its chart debugging PAUSED for item 2 (user
   2026-10-06). User 2026-09-30: "will reduce POI zones in each
   cycle to just 1 (vs. currently where a cycle may have multiple POI zones)". **SPEC DECIDED 2026-10-05** (the user's
   spec, six rounds of questions): ONE IC per cycle, chosen by the *institutional pattern* (initial / opp /
   continuation legs, ranks 1A–3D, tiebreakers) among the §3.1 candidates touching the 61.8–80 zone; no pattern → no
   POI. Canonical: `engine_v2/plans/PLAN_H_institutional_pattern_poi.md` (+ `plans/plan_h_inputs/`: the spec copy,
   figures, prototypes). Implementation plan PLAN_H §8 REVIEWED + folded 2026-10-06 (3 lenses, §8.8; the user's R1–R4 in §6);
   landing A (the selector + debug CSVs `*_ip_candidates.csv`, nothing switched) BUILT 2026-10-06, `/compare` == its
   prediction (PLAN_H §8.3a), saved `20261006_120513_e1c597c`; the variant replays V1 / V2 / V3 + the test census
   MEASURED 2026-10-06 and B's prediction written (§8.4a: V3 == the hand cross-check; the 4 never-active POIs are
   collapsed cycles; T1 delays 2 activations; the census 18 FAIL / 99 moved-silent). **Landing B BUILT 2026-10-06**
   (the switch: one POI per fib record = the pick, MS reads the same selector, the rule-3 meta migration, HV hovers;
   PLAN_H §8.5a): `/compare` vs S_A == the prediction; tests 1258; mutants 32 / 32 killed; chart-reviewed 2026-10-06 — the
   user will DEBUG the charts in a new session (questions, chart aids, rule experiments; each compared against S_B and
   S_A), the points raised are PLAN_H §8.5b; committed `e54f509`, saved S_B `20261006_151106_e54f509`; that
   debugging session (2026-10-06, PLAN_H §8.9): chart aid F1 = the M15 fib layer + the fib hover's
   lifecycle / IP outcome on both charts (the fib draw gate: collapsed hidden, ended never-locked drawn);
   + F1·1 (the fib hovers' literal `%%`); the debugging PAUSED 2026-10-06 (user, PLAN_H §6 F2a) for item 2 — it
   resumes after item 2 with every logged point (PLAN_H §8.5b, §8.9); B's landing review (+ F1) moved to the start
   of the item-2 session, before its plan (F2e; was: deferred until the debugging rules settle) — RAN 2026-10-06c
   (PLAN_H §8.5c, 2 lenses, 520,963 tokens): no BLOCKER / MAJOR; 15 test survivors (worst: the sell-side activation
   touch) + stale docs → ONE fix-up commit (LR1), a static guard on broad `except` scopes (LR2). The V30 / V60 / V90 code is KEPT unwired (X1); POI_ZONES_SPEC §3.2 stays unapplied
   (D4; re-specified by the user 2026-10-09 as the IC reference condition — PLAN_K, D.2 below). Baseline (Week 8 save `20260930_151531_8b11bc2`; one POI row = one IC, carrying 1–3 variants): H1 5 ICs over
   3 cycles; M15 confluence 30 over 19; counter 12 over 7. **What moves when the POIs move** (the user accepted it, D2 /
   C3 — "We won't be able to judge whether those changes are appropriate or correct until I can review the charts";
   PLAN_H §3): H1 sid 1 cycle 2's sd-proximity `CTS_CONFIRMED` at 926 (IC 865 → 832: ≥ 939 or pullback); M15
   confluence sub 1 cycle 1's at 1902 (IC 1889 → 1875: likely survives); the zone-proximity sd triggers H1 sid 0
   cycle 1 710 → 748 and sid 1 cycle 2 926 → 939 — the H1 WVMI gates and counter subs 3 / 5's triggers → 6 of the 11
   trigger records re-derived. Related: "Cascade-viz hover extras" — closed by landing B ("Recently closed").
   **Item 1's rule experiment RX1 (the IC reference condition, PLAN_K) COMPLETE 2026-10-09** (D.2 below). **NEXT — the
   IC identification revamp (user 2026-10-09, the first of the two "immediate future" sessions):** "Additional
   updates around our IC rules and processes. I will provide the longer holistic write-up of the full identifying IC
   process, along with the remaining steps and rules that we have not discussed yet. We can make all the remaining
   updates in 1 go, or we can split them into smaller parts. All that we can discuss. But once all the necessary
   changes / updates are made, we will be done with our identifying IC revamp & update (for now, though there may be
   deferred items we revisit in the future)." Inputs already decided: the IC flow (PLAN_H §6 RX1-F — find IC
   candidates, the conditions filtering together > the IP > fallbacks) and the reasons (RX1-N); waiting on that
   write-up (all DONE by PLAN_L L1 + the decisions below): POI_ZONES_SPEC §3's step structure, the code's
   "candidates" names (`find_ic_candidates`, `*_ip_candidates.csv` → `*_ic_candidates.csv`), the fallbacks (Step 3
   the ranking rules, L2; Step 4 — D.2 "Step 4 of IC identification", deferred). The debugging log's "sub 3 cyc 3
   has no POI record" (PLAN_H §8.9) — CLOSED 2026-10-28 by SESSION ORDER item 9b (PLAN_W §2): its case, a cycle that ends
   before its fib locks, now gets a full POI record (the gate `active OR locked`; 0 on the reference window). **The write-up RECEIVED 2026-10-09** (verbatim
   `engine_v2/plans/plan_h_inputs/ic_identification_revamp_spec.md`; the reading + the measurement: PLAN_H §8.9 "The
   IC identification revamp"): Step 1 the IC candidates, Step 2 the IP (both as built), **Step 3 NEW — the general
   ranking when no pattern exists** (its own four ranking rules over the unfiltered Step 1 list — a separate rule set
   from Step 2's tiebreak rules, PLAN_H §6 ICR-S), **Step 4 DEFERRED — the
   lower-TF fallback** when no IC candidate remains (D.2, rewritten). Decisions PLAN_H §6 ICR-S, ICR-1 … ICR-15 (2026-10-09 → 10);
   trial replay measured (PLAN_H §8.9); **the plan `engine_v2/plans/PLAN_L_ic_identification_steps.md`** (cold-reviewed,
   0 BLOCKER) — two landings: **L1 the restructure + names COMMITTED `b5c3613` + SAVED S_L1 `20261010_114846_b5c3613` 2026-10-10**
   (save `09dffad`, trunk `02a40a7`; = `LATEST_<branch>`, L2's baseline; the user's steps in code + POI_ZONES_SPEC §3,
   the renames, the CSV listing every agreed-opposite candle + 2 columns; the selection identical — `/compare` vs
   S_TBU == the prediction, 31 / 31 mutants killed, suite 1553; PLAN_L §9), **L2 the ranking rules COMMITTED `6d8cc67` + SAVED
   S_L2 `20261010_155816_6d8cc67` 2026-10-10** (save `8aa4f80`, trunk `48ad744`; = `LATEST_<branch>`; the user's chart review "this looks good
   let's proceed"; `/compare` vs S_L1 == the prediction — 7 new POIs, no structure / trigger / sub / WVMI change; suite
   1609, mutants 56 / 56, landing-reviewed, all folded; PLAN_L §10; every label decided 2026-10-10 — PLAN_L §8.2, PLAN_H §6 ICR-15). **First, the IP tiebreaker update (user 2026-10-09, before the write-up — "with
   it, I think we would wrap up the Institutional Pattern process"): COMMITTED `09f056a` 2026-10-09** — a new first tiebreaker, the
   immediate unfilled imbalance (an unfilled sd gap whose first gap candle is at IC + 1 / IC + 2; a filter), then the
   IC's candle type, then the most recent before the imbalance; descriptive `ip_tiebreak` names (no more TB1 / TB2);
   on the window one pick moves (confluence sub 0 cyc 3 1447 → 1660). PLAN_H §6 TBU-1 … TBU-6, §8.9. **Saved S_TBU `20261009_171255_5b5c307`** (source `5b5c307` = `09f056a` + the close-out docs; save commit `0047646`, trunk `2abc143`; reuse of the 44.3 s replay, FETCH GATE PASS) — PLAN_L L1's baseline (`LATEST_<branch>` until S_L1).
2. **The cross-cycle fib revamp — one logic for main and sub, the final cross-cycle + single-cycle fib rules** —
   **COMPLETE 2026-10-09** (PLAN_I S1–S6 + PLAN_J; item 1's debugging resumes next). **OPEN — a cross-limit price
   DEBUGGING task (user 2026-10-09, the second "immediate future" session, after item 1's IC revamp):** "Updates to
   the new cross cycle fib rules we've established using cross limit prices. This is actually an important debugging
   task that is currently causing some subs to have lower cross limit prices than they should, which may affect
   61.8-80 Fib Zone placement and IC candidates. I will provide more details when we get there and on the fix."
   **Details received 2026-10-10 → `engine_v2/plans/PLAN_M_counter_sub_cross_limit.md`** (the write-up verbatim:
   `plans/plan_m_inputs/counter_sub_cross_limit_writeup.md`): a COUNTER sub (its direction opposite its first record's
   parent) takes `P` = that parent cycle's BOS threshold, read live (frozen at that parent cycle's end; a cycle's cross
   permission fixed at its establishment); every other trigger unchanged. Round 1 DECIDED (PLAN_M §3); the trial
   replay = the window's prediction (PLAN_M §4: sub 3 cyc 2 becomes a cross from cyc 1, as H1 sid 1's; its POI's IC
   2969 → 2821; no structure / trigger / sub change); round 2 DECIDED 2026-10-11 (PLAN_M §3 CL-5…CL-8: the source
   `parent_bos_threshold`; the export = the final `P` + start-`P` / move-count keys; the IC bound reads `P` as of each
   moment; the residual below parked); the live-`P` census on 6 windows (PLAN_M §6: the parent's threshold moves
   inside 5 of 14 counter subs' lives; one cycle's permission changes) and the design + two landings drafted
   (PLAN_M §7–§10); cold-reviewed 2026-10-11 (760,821 tok; every finding folded — PLAN_M §11; the export reads `P`
   as of the structure's end, PLAN_M §3 "readings" — DECIDED (a) by the user 2026-10-11); landing 1 (the live-`P`
   plumbing) BUILT + landing-reviewed + COMMITTED `86d4f7b` 2026-10-11 (PLAN_M §12–§13: byte-identical vs S_L2, the
   census unchanged, suite 1657; review 451,068 tok, 0 BLOCKER / MAJOR, all folded); SAVED S_M1
   `20261011_204047_86d4f7b`; landing 2 (the counter-sub source) BUILT + landing-reviewed (PLAN_M §14–§15: = the
   trial's outputs, the census pinned table 14 / 14, 0 knock-on, suite 1691; review 437,943 tok, all folded); the
   user's chart review "charts look good. go ahead" (2026-10-12) → **COMMITTED `10e4c6f` + SAVED S_M2
   `20261012_023137_10e4c6f`** (save `3ebe857`, trunk `475bbf9`) = `LATEST_<branch>`. **PLAN_M COMPLETE 2026-10-12.**
   **Follow-up CL-8 — CLOSED 2026-10-12 with NO change** (the user: "1, keep it — close CL-8 with no change"; the
   difference is correct — sub 3 really tracked a cycle the retroactive sid 1 starts after, and the IC reference rule
   applies as written; PLAN_M §17). **Item 2 has no open work left.** The case, as parked (user 2026-10-11, PLAN_M §3 CL-8):
   sub 3 cycle 2's IC (2821) still differs from H1 sid 1 cycle 1's (732) — sub 3 has one extra cycle before sid 1's
   first, so its cross starts at its own cycle 1 and its IC reference is cycle 0's CTS (0.5818, bound 2824), which cuts
   732's M15 period; sid 1's cross starts at cycle 0 → the far structure side. Detail: PLAN_M §4; measured on S_M2
   (2026-10-12): PLAN_M §17 (no reference change would reproduce 732 — the M15 ranks pick 2837, 1A.iv). [Since
   PLAN_O (2026-10-13, the user's change 2 — a cross's CTS reference = its target's previous CTS): sub 3's cross
   reads CTS_1 0.57902 (bound 2991) and picks 2837; sid 1's cross reads CTS_0 0.57902 (bound 747) and keeps 732.]
   Its reach since PLAN_K: `P` sets the cross window `M` (which cycles a cross may span — the fib's anchors, so the
   61.8–80 zone) AND the IC reference's structure side (the anchor behind `P`, the cross-limit breakout watch's
   break, `T ≤ M`) — a changed `P` moves both. Canonical rules: CROSS_CYCLE_FIB_SPEC §3; PLAN_I §1.2 / §1.4. Was NEXT (user 2026-10-06, PLAN_H §6 F2b–F2e; it replaces the open decision below): "instead of deciding on this, I will
   instead revamp the cross cycle fib rules. Furthermore, my hope is that we can unify the logic across main and sub.
   This update will likely be the final logic we use for cross-cycle & single-cycle fibs." A NEW session: it opens
   with B's landing review (item 1), then decides whether planning and build are separate sessions; the user brings
   the rules. **Split DECIDED 2026-10-06c (PLAN_H §6 LR3): THREE sessions** — (1) the user's rules → the spec (+ cheap
   status-quo measurements), (2) the implementation plan from a fresh code audit + a cold review, (3) the build.
   **The user's rules received 2026-10-07** → canonical `engine_v2/plans/PLAN_I_cross_cycle_fib_revamp.md` (living
   doc; the verbatim text `plans/plan_i_inputs/cross_cycle_fib_spec.md`): a cross LIMIT for main and sub — crosses up
   to the first cycle whose CTS anchor is beyond a reference price set by the structure's starting trigger; rounds 1
   to 5 ANSWERED 2026-10-07 (PLAN_I §6 Q1–Q8, R2-0…R2-5, R3-1…R3-6, R4-1…R4-7, R5-1…R5-3; the rule as decided:
   PLAN_I §1.2 + §1.4; unique subs numbered at their lifecycle START, R4-2; SPEC AGREED, no open question; session 1 CLOSED
   2026-10-07; session 2 (2026-10-07): the implementation plan DRAFTED — PLAN_I §9, stages S1–S5 (incl. S3b, the single's two latent quirks, R6-1b / R7-1) predicted by variant
   replays, cold-reviewed (PLAN_I §9.8); rounds 6–7 ANSWERED (PLAN_I §6 R6-1 … R7-4); the plan APPROVED 2026-10-07; after the build: S6 retires the
   `fib_mode` meta key (R7-2) — LANDED 2026-10-09 (session 7); session 3 = the
   build: S1 + S2 LANDED 2026-10-07 / 08, PLAN_I §9.10; next S3 + S3b + S4, then S5, then S6). **Parked to the S5
   session's first round (R8-1, 2026-10-08):** PLAN_I §9.2 F16 — MS can confirm a CTS with an anchor AFTER its
   confirmation candle (a range back-fill look-ahead; emission ≠ processing order; today's drawn fib locks at the
   earlier anchor; D5's S5 order tripwire would raise) — found by the session-4 random-tail search, 0 on the window;
   and F17's MS side (R8-2): a cycle established on the previous cycle's CTS candle / a BOS anchor after its CTS anchor
   (the downstream's silent skip of such an establishment is made LOUD now — S2b, `FibEventOrderError`).
   **Also parked (S4 landing review, 2026-10-08):** PLAN_I §9.2 F19 — a fib outlives its cycle when the next cycle
   gets no fib (`_current_cycle` + set-if-absent terminals; latent, 0 on the window). **DECIDED 2026-10-08 (round 9,
   PLAN_I §6):** R9-1 (b) — S5 lands first (its order tripwire raises on F16), THEN an **MS fix plan** (F16: a
   back-filled candle's re-step sees the state as of that candle; + R9-2 (a), F17's MS side: a breakout must break
   cycle T−1's confirmed `cts_threshold` at its apply, the BOS window ends at the CTS anchor) — its own plan and
   `/compare`, NEXT after S5 (+ S5b); R9-3 (a) — F19 fixed in stage **S5b** (the cycle-table end caps a fib's end) — **S5b LANDED 2026-10-08**
   (byte-identical on the window, PLAN_I §9.10); next the MS fix plan = **`engine_v2/plans/PLAN_J_ms_restep_state_and_new_cycle_break.md`**
   (round 1 answered 2026-10-09: J1a, J2b) — **J3, J2, J1 LANDED 2026-10-09** (local; PLAN_J §9). Open follow-ups
   there: J3's swap → a raise LANDED 2026-10-09 (session 7; byte-identical, PLAN_J §9); **S6 LANDED 2026-10-09** (session
   7; the `fib_mode` meta key / fib CSV column retired, format-only, PLAN_I §9.10) — **the revamp is COMPLETE**. Baseline: **S_S6** `20261009_020650_5f6bae3` (saved 2026-10-09,
   = `LATEST_week9-entries-decisions`; before it S_J `20261009_013718_708a8d9` (PLAN_J), S_F1 `20261006_224610_42c137c`). Likely inputs (to confirm with the
   user): the 2026-10-01 item and D5 below, and §E "Cross-cycle fib bundled follow-up" with its parked open
   questions (`engine_v2/zones/CROSS_CYCLE_FIB_SPEC.md` §13).
   Was: **Single-cycle fib POIs on subs, and revisit the overall sub cross-fib logic** — user 2026-10-01: "Single-cycle fib
   POIs on sub, and also additionally revisit the overall sub cross-fib logic". Today a sub's single-cycle fib is only
   a fallback (cross fails, established phase); the change lets the two coexist — additive, more POIs, so it pulls
   against item 1, which is decided first. Iterative: scenario / parameter flips + chart review. **Also fix here
   (user 2026-10-02, PLAN_H D5): the sub MS snapshot's fib — DONE by PLAN_I S5 (2026-10-08: MS reads its embedded
   FibTracker's decision, CROSS_CYCLE_FIB_SPEC §8)** — a sub's in-flight POI snapshot (sd-proximity CTS
   confirmation) always uses the INTRA-cycle fib (`select_fib_anchor_for_cycle` → `"intra"` for sid 0, every sub)
   while the drawn sub POIs come from the cross fibs: they differ on 8 unique sub cycles (e.g. confluence sub 0 cyc 2:
   drawn POI IC 678 vs MS's fib zone ~100 pips away); no outcome diverged on the window. "I think this IS something we
   do need to fix, but we can do that as item 2." Detail: memory
   `project_sub_single_cycle_fib_pois.md`; `engine_v2/zones/CROSS_CYCLE_FIB_SPEC.md` (§13), `POI_ZONES_SPEC.md`
   "Cross-Cycle Mode".
3. ~~**Zone size adjustments**~~ — **MOVED 2026-10-27 to D.3** (Parts 2 / 3: the adjustment goes into the entry price
   and stop loss, not into the zones — the user, after item 5).
4. **Pinbar body-pip floor calibration** — **DONE 2026-10-28 (SESSION ORDER item 7, the user 2026-10-27: "the last major item I
   want to tackle in the next session")**. Scheduled here by the user 2026-10-01 (was TRIGGER: strategy optimization):
   the provisional values (H1 2.2 / M15 1.0 / M5 0.5 pips, NZD_USD only) and the reclassify-to-pinbar approach itself.
   Iterative: parameter flips + chart review; a candle-type change cascades (patterns → structure → zones), so every
   flip gets a full `/compare`. **2026-10-28 — DONE, PLAN_U** (`engine_v2/plans/PLAN_U_rolling_pinbar_body_floor.md`,
   the canonical build spec): the user chose a ROLLING floor (the p90 of the bodies of the 90 most recent pinbars
   strictly before the candle, the history fetched before the frame; both windows tried — body-% pinbars only vs + the
   floor-demoted ones; the fixed per-TF table removed). Three landings: U1 the per-candle floor plumbing + the three
   columns + the candle hover on today's fixed floor — LANDED 2026-10-28, format only; U2 the rolling floor, natural
   window — LANDED 2026-10-28, S_U2 `20261028_113834_84fbbc6` (the fixed table removed; = the prediction: H1 74 / M15 294 candles become pinbars; no
   sub, trigger or POI pick moves — PLAN_U §9 "U2"); U3 the final window (the floor-demoted pinbars fill the window
   too) — the user's pick after seeing both, natural kept as a switch — LANDED 2026-10-28 (= the prediction: vs U2 H1
   15 / M15 64 candles change type, four downstream records, no sub / trigger / POI / IC pick — PLAN_U §9 "U3"); U3b
   the seed guard steps back until the full and the drop-one-window final runs end on the same window (the check as
   first worded fails on the 2-window seed — measured; the user: "Guard extends") — LANDED 2026-10-28 (= the
   prediction: H1 3 steps / 840 candles / 355 body-% pinbars, M15 3 / 816 / 310; every output byte-equal to S_U3 —
   PLAN_U §6b / §9 "U3b"). Still provisional: the window / percentile (90 / p90) — CANDLE_BODY_FLOOR_NOTES
   "Open / may-revisit". The decisions verbatim:
   `engine_v2/features/CANDLE_BODY_FLOOR_NOTES.md` "Direction — the rolling floor". Detail: memory
   `project_pinbar_body_pip_floor.md`.

**D.2 — Backlog: decided after D.1 is complete (user 2026-10-01).**
- ~~**Entry confirmation candle patterns**~~ — **MOVED 2026-10-27 to D.3** (Part 2 — the user: "We will also move
  entry candle patterns to part 2 of Week 9, thus allowing us to complete part 1").
- ~~**The M5 layer**~~ — **MOVED 2026-10-28 to D.3** (the user, item 8's scoping: built when Part 2 needs M5). Was:
  (`H1.main >> M5.counter` — a third lens directly under H1.main, the Part 4 vision's 4th chart;
  memory `project_part4_vision.md`) — DECIDED 2026-09-30: its own Heavy block (plan + cold review), before any Entries
  work that reads M5; not a Week 8 blocker. Footprint seen so far: `debug/export_sub_tables.py` hard-codes `_M15_`, and
  `pipeline/orchestrator.py` (~:646) runs the whole multi-TF path only when `"M15"` is in `lower_timeframes`. Related
  (only if M5 ever nests under an M15 sub): the canonical encoding of a deeper path, e.g. `H1.main >> M15.counter >>
  M5.confluence` (PART4 ~:176 "Open (deferred)"; PART4 §13.6, §17.12). BACKLOG.
- ~~**Zone strength scoring**~~ — **MOVED 2026-10-28 to D.3** (the user, item 8's scoping: with zone size at entry and
  stop loss). Was: syllabus Week 7 (`strength_score` / `strength_flags`, "strong zones look strong", click a
  zone → strength + flags); `KLZone.strength` is always `0.0` (`zones/kl_zones_v1.py`). DECIDE (scope). Idea attached:
  scale strength thresholds by ATR percentile (volatility regimes).
- ~~**Candle-direction / rejection confirmation at a zone touch — as an ENTRY condition**~~ — **MOVED 2026-10-28 to
  D.3** (the user, item 8's scoping: Part 2, next to the entry candle patterns). Was: (user 2026-09-30: not a trigger
  filter; triggers stay structural). The May trigger filter (`47e7532`, reverted `2afd3b2`) is the reference form:
  trigger candle body opposing the zone side. DECIDE when Entries are designed. Detail: this register's "Recently
  closed" (the 2026-09-30 measurement).
- ~~**A same-candle replacement chain names a replacer that never starts**~~ — DONE 2026-10-28 (SESSION ORDER item
  8b, PLAN_V §2; 0 on the window — zero output change): `ended_by` = the replacer that survives the candle — phase 3
  takes the LAST of the equal-priority end entries, and phase 1 re-points a co-starter frozen at the candle when its
  freezer is frozen there too. Was (PLAN_I S1's landing review, 2026-10-07; latent since Plan C): the FIRST entry, so
  the incumbent named the replaced co-starter — a sub that never starts (empty `ended_by` + `WARNING [sweep] record …
  ended by pool#k, which never started`). Pins: `test_lifecycle_sweep_unit.py::test_a_same_candle_replacement_chain_
  names_the_replacer_that_survives` (was `…_leaves_ended_by_empty_and_warns`, which pinned the old behaviour),
  `test_three_co_starters_all_name_the_last_one`, `test_the_re_point_leaves_a_record_ended_at_an_earlier_candle`.
- ~~**The fill check's label-based paths disagree on a frame with a missing label**~~ — DONE 2026-10-28 (SESSION ORDER
  item 9a, PLAN_W §1; zero output change): the user's pick "Require 0..n−1" — both fill checks raise `ValueError` on a
  frame not indexed 0..n−1 (`common/types.require_positional_index`), the label-based fallback loop and the cache's
  label → position dict deleted (the census: 188,013 of 188,013 `is_filled` calls on a 0..n−1 frame —
  `plans/plan_w_inputs/w9a_index_census.txt`). Was (PLAN_S step 2's landing review, 2026-10-21): `is_filled`'s
  fallback skipped a missing label (`continue`; a `break` mutant survived the suite) while the cache returned
  `(None, None)` when the label `end_idx + 1` was missing. Pins: `test_imbalance.py::test_the_fill_checks_raise_on_a_
  frame_not_indexed_0_to_n_minus_1`, `test_the_fill_checks_accept_an_int64_index_0_to_n_minus_1`.
- ~~**Should subs get the DERIVED CTS KL zone as their probe reference?**~~ — DECIDED 2026-10-28 (SESSION ORDER item
  9c, PLAN_W §3): the user: "Derived, as of the read" (9c-2: a CTS_CONFIRMED winner's own sub's CTS KL zone as of
  the read — was ad hoc, = the derived zone's INIT step; 1 of 11 reads differs on the window, no start moves) and,
  for exporting / charting them, "Export + chart, toggle off" (9c-1). Detail: PART4 §17.8, GOTCHAS (the `kl_zones=[]`
  pool-path note).
- **Step 4 of IC identification — the lower-TF fallback — DEFERRED, possibly never built** (was "No-pattern cycles —
  a fallback POI"; DECIDED by the user's IC identification write-up 2026-10-09, `engine_v2/plans/plan_h_inputs/
  ic_identification_revamp_spec.md`, PLAN_H §6 ICR-*). A no-pattern record gets its IC from Step 3, the ranking
  rules (D.1 item 1, PLAN_L L2, 2026-10-10 — POI_ZONES_SPEC §3.6). What remains: when NO IC candidate is left — Step 1 finds none (`no_ic_candidate`;
  since PLAN_P, 2026-10-13, after Step 1A — the 80–100 fallback zone — found none either)
  or ranking rule 1 removes every candidate as a pinbar (`all_ic_candidates_pinbar`) — run Steps 1–3 on the lower TF's
  data (the user's spec: "in future iterations we will build an additional process where we would go into a lower TF";
  why the leg cap is 3 on H1 / M15 and likely 2 on M5; PLAN_H H2). The user: "This is actually something I want to
  defer for now because it could become quite complex, but may not give us much benefit if most of our cycles are able
  to find an IC using Steps 1-3 … We may also ultimately choose to skip Step 4 entirely and decide that not every cycle
  MUST have an IC & POI zone." Until then: no IC → no POI. The kept V30 / V60 / V90 selection (PLAN_H X1) was the
  other fallback candidate; Step 3 takes that role (L2), the code stays unwired ("Keep it for now", PLAN_H §6 ICR-5).
  On S_TBU: 1 record would reach Step 4 (sub 1 cyc 2, `no_ic_candidate`). Since PLAN_P (IC condition 3 = the
  immediate gap + Step 1A): 7 of the window's 26 drawn records — all-pinbar in 61.8–80 (H1 sid 0 c0, sid 1 c0, sid 1
  cross 0→1; sub 3 c0 — Step 1A does not run after Step 3), all-pinbar in 80–100 (sub 2 c0), none in either zone
  (sub 1 c0, c3); the user (2026-10-13): "For now it is" (no POI). **KEEP DEFERRED (the user 2026-10-28, item 8's
  scoping):** decide once Part 2's entries show whether the records without a POI matter (the 7-of-26 count is from
  PLAN_P, 2026-10-13 — re-measure first). DECIDE (whether to build it at all).
- **Test the POI_ZONES_SPEC §3.2 scenario IC filters — COMPLETE 2026-10-09 (RX1 = PLAN_K landings A + B; the final save S_K_B `20261009_140048_97795bd`, `LATEST_<branch>` until S_TBU `20261009_171255_5b5c307`)** — never implemented; kept in the spec, unapplied (PLAN_H D4:
  "I still want to keep these and test into them in the future"). Any IC filter stays OUTSIDE the institutional pattern
  and is applied first. **TAKEN UP 2026-10-09 as item 1's first rule experiment, RE-SPECIFIED by the user for today's
  fibs** (PLAN_H §8.9 RX1, decisions §6 RX1-*; verbatim `plans/plan_h_inputs/ic_reference_conditions_spec.md`): one
  bundled condition — the IC entirely on the near side of the level the cycle broke out of, and before the pattern that
  broke it (+ a new cross-limit breakout watch). Rule DECIDED; its plan `engine_v2/plans/PLAN_K_ic_reference_condition.md`
  (cold-reviewed, K-Q1…K-Q8 answered); **landing A BUILT 2026-10-09** — computed + exported (the IP CSVs'
  `ic_ref_reason`, the fib meta `ic_ref_*` + hover), the pick unchanged (`poi_zones._IC_REFERENCE_FILTER = False`);
  POI_ZONES_SPEC §3.2 rewritten (the old tables → its Appendix A); saved S_K_A `20261009_112137_02e9381`. Landing
  review 2026-10-09 (PLAN_K §11.2, 651,669 tokens): LA1 FIXED (the final reference = as of the lifecycle cap, the
  watch on the whole frame), 9 test gaps pinned; the variant replay V measured (PLAN_K §11.3). **Landing B BUILT
  2026-10-09** (PLAN_K §11.4: applied in both consumers + the `IP: none` reason `no_ic_candidate`, the user's PLAN_H
  §6 RX1-F / RX1-N; `/compare` vs S_K_A == V + the sub 1 cyc 2 relabel) — chart-reviewed + committed 2026-10-09
  (not saved). OPEN (latent, 0 on the window; PLAN_K §11.2 B5): can MS confirm a BOS on a reversal's own apply candle?
  Both reversal builders take the most recent BOS_CONFIRMED with moment ≤ the reversal (inclusive, pinned) — such a
  BOS's threshold would not be `P`, so its anchor would not be "the anchor behind `P`". DECIDE only if reachable.
  PARKED by the user 2026-10-28 (shown it as an item 8 leftover): "I think this case is not possible, so let's park
  for now".
- **The probe's ad-hoc BOS_0 zone reads past its search bound ("L5b")** (found by Plan A 2026-09-19, "deliberately
  open"; registered by the item-10 sweep 2026-10-28): at a reset candidate `unified_probe._bos0_inner_at_start` →
  `build_ad_hoc_bos0_reference_zone` → `identify_base_pattern` reads up to 5 candles past `probe_end_idx` — a
  reproducibility gap against the stated bound, not live look-ahead; closing it "can move the H1 sid-1 start through
  the main reversal probe" → its own decision + `/compare`. Never measured. DECIDE. Detail: LANDMINES "Probe
  `end_idx` Is the Supreme Bound"; PLAN_A §8.
- **`REVERSAL_CANDIDATE.meta["pattern"]` is always `"?"`** (found 2026-09-19; registered by the item-10 sweep
  2026-10-28): `market_structure.py` `_schedule_reversal_from_anchor` (:811) and the `[RV_APPLY]` line (:848) read
  `.pat` / `.pattern`, the name is `.name` — all 6 candidates on S_W9d (H1 1, confluence 5). Meta only (no decision
  reads it; STATE_CHANGED's `meta["pat"]` is right); a one-line fix with its own `/compare` (the 3 events CSVs).
  OPEN. Detail: GOTCHAS "`REVERSAL_CANDIDATE.meta["pattern"]` Is Always `"?"`".
- ~~**The agreed direction for wave candles + WVMI**~~ (PLAN_H Q-W) — DONE 2026-10-28 (SESSION ORDER item 8c, PLAN_V §3):
  `wave_candles._is_qualified` = the agreed direction == the required one, so a volume-only doji qualifies (the user,
  after the variant replay: "Apply it"); a wave candle on a direction-0 candle is drawn in its `vol_dir` colour on
  every chart ("Draw it in its vol_dir colour"). On the window: 0 of 53 wave candles move; sub 3's WVMI temporary-LP
  history gains M15 2907 / 2913 / 3252 (+3 rows per lens `*_wvmi_lps.csv`, +3 sell dots per M15 chart). Was: the IP
  legs got the agreed direction in 2026-10 (item 1), wave candles + WVMI kept "qualified" — user 2026-10-05: "we
  likely will also make the change for wave candles & WVMI, but we'll come back to that later"; one helper since
  2026-10-26 ("WVMI back on the chart" step 3).

**D.3 — Moved out of Part 1 into Parts 2 / 3 (the user 2026-10-27, after item 5; SYLLABUS Week 9).**
- **Zone size at entry and stop loss** (was D.1 item 3 / SESSION ORDER item 6, "zone size adjustments") — the zones
  stay in their base form; the size adjustment becomes part of the entry decision: the entry price (likely near the
  zone's inner edge) and the stop loss (around its outer edge) are adjusted when a zone is too big or too small. Why
  (the user): changing the zones themselves "would have additional consequences such as altering proximity
  triggers". The user's rationale verbatim: SYLLABUS Week 9 Part 2 "Zone size at entry and stop-loss". Background:
  the user 2026-09-30, "currently some KL & POI zones may be very large or small in their current base form"; the
  measured sizes: memory `project_week9_part1_inputs.md`; the zone specs `engine_v2/zones/KL_ZONES_SPEC.md`,
  `POI_ZONES_SPEC.md`; the Naming-Standard audit (PLAN_N, 2026-10-12) landed for it (a KL step's index is `base_idx`
  on INIT, `expanded_idx` on an expansion — `kl_zones_v1.bounds_step_draw_idx`). Related: "Zone strength scoring"
  (D.2). OPEN (Part 2 / 3; the user's spec).
- **Entry confirmation candle patterns** (was D.2) — Part 2 (the user 2026-10-27: "We will also move entry candle
  patterns to part 2 of Week 9, thus allowing us to complete part 1"). User 2026-09-30: "define new candle patterns
  that will be used to confirm entry (when necessary). This is a prerequisite for Entry, but can go last in Part 1 or
  first in Part 2." Related: D.2 "Candle-direction / rejection confirmation at a zone touch"; D.1 item 4 (the pinbar
  floor — the patterns read candle types); the pattern engine (`patterns/structure_patterns.py`,
  `features/candles_v2.py`). OPEN (Part 2).
- **Candle-direction / rejection confirmation at a zone touch — as an ENTRY condition** (was D.2; MOVED 2026-10-28,
  the user at item 8's scoping) — Part 2, next to the entry candle patterns. User 2026-09-30: not a trigger filter;
  triggers stay structural. The reference form: the May trigger filter (`47e7532`, reverted `2afd3b2`) — the trigger
  candle body opposing the zone side; the 2026-09-30 measurement: "Recently closed". OPEN (Part 2).
- **Zone strength scoring** (was D.2; MOVED 2026-10-28) — with "Zone size at entry and stop loss" above. Syllabus
  Week 7 (`strength_score` / `strength_flags`, "strong zones look strong", click a zone → strength + flags);
  `KLZone.strength` is always `0.0` (`zones/kl_zones_v1.py`); idea attached: scale the thresholds by ATR percentile.
  DECIDE (scope). OPEN (Part 2 / 3).
- **The M5 layer** (was D.2; MOVED 2026-10-28) — built when Part 2 needs M5: its own heavy block (plan + cold
  review), before any Entries work that reads M5 (decided 2026-09-30). The footprint and the deeper-path encoding: the
  struck D.2 entry. OPEN.

## E. Trigger-based (only when the trigger fires)

- **Cross-cycle fib bundled follow-up — CLOSED by PLAN_I (S4 + S5, 2026-10-08):** main runs the one FibTracker logic
  (every cycle, the full walk, fill-as-of "current"; the bespoke main meta migrated, Q8), MS reads the same decision
  through its embedded tracker (S5 — the cycle-1-only resolver retired), the Scenario-1 / revert ↔ ceiling merge is
  moot, `M = 0` is pinned (CROSS_CYCLE_FIB_SPEC §9). Was — TRIGGER: a window that produces a MAIN cycle ≥ 2 cross. Then together:
  generalize the MS in-flight POI resolver (`compute_poi_inners_for_cycle`, cycle-1-only) to the full walk ("11b
  Part B"); reconcile the main cross's bespoke meta; widen the config window incrementally to FIND such a window and
  chart-review + `/compare` it. Parked with it (open questions): fill-as-of convergence main → "current"; the
  Scenario-1/revert ↔ ceiling merge; `M = 0` behaviour; the deeper-walk fill-as-of decision. Detail:
  `engine_v2/zones/CROSS_CYCLE_FIB_SPEC.md` §11b + §13, memory `project_cross_cycle_fib_unification.md`.
  2026-10-06: likely absorbed by §D.1 item 2 — CONFIRMED (the revamp COMPLETE 2026-10-09: PLAN_I S1–S6 + PLAN_J).
- **Plan F §7 follow-ups** — re-ask a failed fib decision when a relevant gap's c3 closes ("dropped" → "one candle
  later"); ~~the MS vs FibTracker activation divergence~~ (retired by PLAN_I S5 — MS reads FibTracker's decision;
  found stale by the item-10 sweep 2026-10-28); latent: `inst.meta` mutated on imbalance
  instances shared across projections, the chart hover picks a POI by nearest inner price, single-candle stretches
  dropped. TRIGGER (chart review shows a dropped fib) / parked by the user. Detail:
  `engine_v2/plans/PLAN_F_imbalance_c3_knowability.md` §7.
- **Double RANGE_STARTED from the offline finalize** — fixture-only, 0 on the reference window. Parked by the user.
  Detail: memory `project_zones_timing_audit_20260922.md`.
- **`REVERSAL_CANDIDATE` straddling a sub's cap** — `knowable_at_idx` keys it on `ev.idx`, not its `apply_idx`, so a
  sub capped by something else (parent end, same-direction replacement) could keep a candidate that applies past the
  cap. 0 on the reference window (2026-09-30). TRIGGER: a window where one is kept with `apply_idx > cap`. Detail:
  PART4 §17.12.
- **Performance** — TRIGGER: runtime hurts. Entity-direct MS compute (no slice + lookback; needs an MS refactor —
  LANDMINES "slice-elimination is a deferred optimization"), same-TF feature sharing by reference (PART4 §9.6),
  parallelism and the remaining pandas reads in MS (memory `project_ms_optimization_opportunity.md`).
- **Mirror initializes new columns with `pd.NA`** instead of `_ensure_output_cols`' per-column defaults — TRIGGER: it
  bites (the slice-copy drop is the safeguard). Detail: LANDMINES ("Slice Copies Inherit Mirrored Structure Cols").
- **Audit positional-index df columns that survive the sub slice** — TRIGGER: adding any df column whose value is a
  positional index. Detail: memory `project_is_range_pollution_fix.md` "Open follow-up".
- ~~**Opposite-direction POIs**~~ — CLOSED 2026-10-20 by PLAN_R step 2: the parent's opp_sd proximity ladder reads a
  counter sub's POI (the sides assigned by the sub's direction relative to the parent; a structure's own POIs stay
  sd). Detail: `engine_v2/zones/POI_ZONES_SPEC.md` "POIs are always sd-direction (invariant)".
- ~~**The start-candle probe's market structure runs at the H1 proximity values on M15**~~ — CLOSED 2026-10-27 by
  PLAN_T T1 (the first_confluence probe's MS gets the caller's timeframe values, pip size and its sub's cross-fib
  limit — byte-identical on the window; the TRIGGER below fired with PLAN_T T2, which lets every cycle 0 but two
  confirm by proximity; the user: "I do think that we need to update to 5/30 and be consistent"). Was: (8 pips and
  the 50-pip
  narrow-cycle gap instead of 5 / 30): `unified_probe.py`'s Phase 2 builds it through
  `structure_engine._make_market_structure` without `timeframe` (that helper's docstring: "a deferred cleanup").
  INERT BY DESIGN — the probe's result reads two CTS_ESTABLISHED, cycle 0's CTS_CONFIRMED and a reversal, never a
  proximity confirmation, and cycle 0 cannot be proximity-confirmed (`unified_probe.py:549-552`, PLAN_I §9.2 F7).
  TRIGGER: the probe ever reads a proximity-confirmed CTS → pass the sub's `timeframe`. Registered 2026-10-19 (the
  user: "We can add it"); detail `engine_v2/plans/qa_opp_sd_zones_inputs/qa_findings.md` rounds 5–6.
- **An end-to-end pin: an unblocked first_confluence probe whose cycle 0 confirms by proximity reaches its 2nd CTS where
  its sub does** (descoped from PLAN_T T2, §5): no synthetic fixture makes the probe's own cycle 0 proximity-confirm
  (searched 2026-10-27); the agreement holds by construction (one flag evaluation → the probe and its geometry; the same
  MS values since T1), each link is pinned, the log tripwire is unit-pinned. TRIGGER: a fixture (or a window) where it
  happens — or the probe/sub tripwire prints → add the pin.
- **The probe/sub consistency tripwire misses a probe confirming cycle 0 BEFORE its sub** (`no_retrace` branch,
  `entity_df_mutation.check_probe_sub_consistency`): the finalize is the probe's cycle-0 confirmation or its probe
  end, and the check does not know the end. TRIGGER: a reason to doubt a first_confluence sub's lifecycle start →
  hand the check the probe's `probe_end_idx` (in the pool's `ProbeCacheEntry`) and compare exactly.

- **A bounded MS run sees the full-frame imbalance list** (Plan A 2026-09-19, narrowed by Plan F 2026-09-24;
  registered by the item-10 sweep 2026-10-28): `df.attrs["imbalances"]` is built on the whole frame, so a run ending
  at B sees a gap whose c3 is B+1; only a prefix-vs-merged degeneracy mismatch can reach an output — 0 cases on the
  reference data; sub builds immune. TRIGGER: the property test trips on it → record and decide. Detail: LANDMINES
  "Bounded MS Runs Must Not Read Past `end_idx`" (Known residual).
- **The probe's iteration cap is H1's on every TF** (PART4 §15 "Open Items"; registered by the item-10 sweep
  2026-10-28): `unified_probe` gives up at `max_iterations = 10` (all resets → `pending`), calibrated for H1; the M15
  probes use it too. S_W9d: 24 probe runs, ≤ 4 iterations each, none `max_iterations`. TRIGGER: a probe finalizes
  `max_iterations`, or the M5 layer's probes.
- **The pinbar body floor stays provisional** (CANDLE_BODY_FLOOR_NOTES "Open / may-revisit"; was only inside the
  closed D.1 item 4 — registered by the item-10 sweep 2026-10-28): the 90 / p90 window, the reclassify-to-pinbar
  approach itself, the `one_maru_opposite` / `one_maru_continuous` interaction. TRIGGER: strategy optimization (Week
  10). Detail: PLAN_U §7.

## F. Week 10 — live readiness

- Incremental `is_filled` (per-instance fill state machine) — IMBALANCE_FILL_SEMANTICS "When to revisit";
  FIB_LIFECYCLE_SPEC §13 (live per-candle fib evaluation).
- Incremental imbalance detection (instances created at `formed_at`, grown by explicit extension events) — GOTCHAS
  (the imbalance-prefix note, "When to revisit").
- Per-candle (not snapshot) POI inners in MS — MARKET_STRUCTURE_SPEC (~:350).
- Phase 3 per-candle dual-lens driver (the sweep's step body is its loop body) and live-mode pool GC (evict when the
  parent ended + no live reference) — PART4 §17.12.
- Ad-hoc zone wait-for-candles (a zone whose neighbour candle is past the live edge) — memory
  `project_ad_hoc_zone_wait_for_candles.md`.
- Moment-order event processing + live-mode pattern selection — PLAN_E Q3; memory
  `project_zones_timing_audit_20260922.md` item (b).
- A live / incremental probe (candle-by-candle) — `engine_v2/plans/PLAN_B_double_cts_early_stop.md` §7.
- The WVMI batch `update_temporary_lp` pass is dead in batch mode (creation already searched the same bounded range)
  — it only matters live. Detail: `engine_v2/plans/plan_e_inputs/review_scripts/mutants_plan_g.py` (equivalent N11 /
  N12).

## G. Idea backlog (no slot yet)

**Every chart / hover line below is LONG-TERM PARKED** (the user 2026-10-28, at the item-10 sweep, verbatim: "The
overall charts look good for now. We can consider any remaining charting items mentioned in the documentation to be
"long term" parked. These we may not come back to unless I specifically mention something about doing another
detailed sweep.") — the chart revamp, the dot-hover quirks, the hovers not on the conventions, M15 elements on H1,
3-tier opacity, the stepped proximity-reference line, the breadcrumb header. Do not raise them unasked.

- **Fib storage unification** (`_cross_cycle_data` → versioned `_fibs`) — "DEFERRED, not closed (user 2026-05-27):
  may revisit". Detail: `engine_v2/zones/FIB_LIFECYCLE_SPEC.md` §13.
- **Overall chart revamp & improvements** (the user, 2026-10-09: "we can make a note of it and come back to it later
  when we do the overall chart revamp & improvements") — no slot yet; collect chart items here. (1) **The fib hover
  masks candle hovers:** each drawn fib (H1 and M15) carries an invisible hover line at the midpoint of its 61.8–80
  band with one point per candle from the BOS anchor to the CTS anchor (`export_plotly` ~:2579 / `export_m15_chart`
  ~:1700, PLAN_H §8.9 F1); the layout keeps Plotly's default "closest" hover mode, so a candle crossing that price
  inside the span shows the fib instead (H1 sid 1 cyc 2: 0.57816 over 80 candles, through the 832–866 IC area).
  Options raised: hover points only at the band's two ends (recommended), one point at the CTS end, a legend toggle.
  Related: "Uniform 3-tier chart opacity" below; register §A's charting-audit style items. (2) **The proximity
  markers' `zone_confirmed_idx` means two things** (PLAN_Q plan review, 2026-10-14): a KL marker shows the zone's
  lifecycle-CLAMPED `confirmed_idx` (a collapsed cycle: the structure's lifecycle start, after the trigger — 902 at
  the 820 / 871 markers), a POI marker the UNCLAMPED activation floor (PLAN_Q). Option: show the KL zone's unclamped
  confirmation moment (`KLZone.source_time`'s candle) — a second cause, not done in PLAN_Q. Detail: CHARTING_SPEC
  §12.
- **Two names for a cycle's end at the next cycle: the fib's `new_cycle` vs the zones' `next_cycle`** (flagged to
  the user 2026-10-23 with the fib-hover spec, not asked): the fib stream's terminal (`fib_tracker._set_terminal`) is
  `new_cycle`, the cycle table's (`structure_lifecycle.compute_cycle_lifecycle`, which every KL / POI inherits and
  which caps a fib when strictly earlier — CROSS_CYCLE_FIB_SPEC §2) `next_cycle`; since the zone and fib hover specs
  both show on one chart as `end_reason=`. Exported values, so a rename is a meta-contract migration (LANDMINES
  "Event Contract Rules" rule 3), not a hover change. Detail:
  `engine_v2/plans/qa_chart_edits_inputs/user_spec_20261023_fib_hover.md`.
- **Per-LP WVMI rows for COLLAPSED records** — not now; the user 2026-10-26 (before "WVMI back on the chart" step 4):
  "for now collapsed records don't need to get per-LP rows. I may change my mind later". So `*_wvmi_lps.csv` = one
  row per dot (the non-collapsed records' temporary + locked LPs), while `_wvmi.csv` keeps collapsed records with
  `cycle_collapsed=True` (reference window: H1 sid 1 c0 / c1; sub 3 c0 on both lenses; confluence sub 0 c0, sub 1 c0 /
  c1). Revisit only if the user asks. Detail: `engine_v2/plans/qa_chart_edits_inputs/user_spec_20261024_wvmi_on_chart.md`
  round 10.
- **Uniform 3-tier chart opacity** (brightest = cycle not ended, mid = sid not ended, faded = sid ended; one
  `opacity_for(sid, cycle)` helper for every element) — its own session. Detail: FIB_LIFECYCLE_SPEC §9.4 + §13.
- **Breadcrumb header with cross-chart navigation** — "not in v1". Detail: PART4 §16.9.
- **Split `PART4_REFACTOR_SPEC.md` into canonical spec files** (a new `MULTI_TF_SPEC.md` + sections moved into
  `MARKET_STRUCTURE_SPEC` / `CHARTING_SPEC` / `LANDMINES`) — the drafted §13.7 plan; PARKED as optional by the user
  2026-09-30 (§13.7 was finalized in place instead: ~235 `PART4 §x` citations, no behaviour gained). Detail: PART4
  header "Canonical" + footer.
- **The stepped "current proximity reference" line on the H1 chart** (the user, 2026-10-20: "Decide later") — one
  stepped trace per overlay-driven cycle and side (sd / opp_sd) showing the proximity reference in force at each
  candle, its hover naming the zone (parent BOS / POI / CTS, or `sub_N cC KIND`) and the gap; a SAMPLE renders
  from a replay with `engine_v2/plans/qa_opp_sd_zones_inputs/build/prox_ref_sample.py` (reads the overlay's
  `ref_log`). Figure-only if built (PLAN_R §9.8).
- **Would-have-been subs for collapsed parent cycles** (the user, 2026-10-19: "OK we can go with parent zones only
  for now and consider subs later"): the opp_sd zones proximity overlay (SESSION ORDER item 3, PLAN_R step 2, landed 2026-10-20)
  reads sub zones only on parent cycles with a real-time lifecycle; a collapsed cycle (lifecycle floor ≥ end — on
  the reference window structure 1's cycles 0 / 1, `[902, 902)`) scans its parent zones only, because the pool
  builds no subs there (LANDMINES "Collapsed Parent Cycles"), and a live sub of the real-time parent cycle is NOT
  borrowed (decided). The engine's would-have-been clock is the event MOMENT (`ev.idx`); subs in that clock would
  need probes / records outside the real-time pool. Detail: `engine_v2/plans/qa_opp_sd_zones_inputs/qa_findings.md`
  rounds 4–6.

- **The BOS / CTS dot hover quirks (5)** (the user 2026-10-21: "Let's come back to it later to address them in more
  detail"; registered as its own line by the item-10 sweep 2026-10-28): the PB `cycle_id` placeholder 0; PB = a
  pullback or a replacement break; `idx` = the anchor (the confirmation moment / method / pattern not shown, `phase`
  judged at the anchor); `time_1H` never shown, `layer=recent` constant; the M15 overlay's unconfirmed CTS label.
  Detail: `engine_v2/plans/qa_chart_edits_inputs/user_spec_20261022_dot_hover.md` "Quirks".
- **The hovers not yet on the hover conventions** (registered by the item-10 sweep 2026-10-28): the OHLC candle (but
  its body-floor line), volume / EMA, the range top / bottom lines, the reversal candidate, the state / `bo` labels,
  the structure-pattern markers. Detail: CHARTING_SPEC "Hover label conventions".
- **M15 POIs / fibs / confluence zones on the H1 chart** (CHARTING_SPEC §11 "Not yet rendered on H1 (future work)";
  registered by the item-10 sweep 2026-10-28) — each a `subordinate_overlays` key + a render block + style keys.
- **Plan E's optional big renames** (PLAN_E_inputs §2.8, "deferred", 2026-09-24; registered by the item-10 sweep
  2026-10-28): `probe_end_idx`, `creation_event_idx`, `enforce_cts0_new_extreme`, the test constant `CONF`.

## H. Deferred indefinitely

- `armed_idx` chart surfacing; consumer-specific fill predicates (IMBALANCE_FILL_SEMANTICS "When to revisit").

## I. Needs confirmation (possibly superseded)

- (none — the last item, PART4 §15 "Validation strategy", was closed 2026-10-28; see "Recently closed")

---

## Recently closed (history pointers)

- **PART4 §15 "Validation strategy"** (§I; registered by the item-10 sweep 2026-10-28) — CLOSED 2026-10-28 as
  superseded (the user: "Close it"): the suite (2,173 tests incl. the pool / sweep / registry / per-entity suites),
  the mutant runs and `/compare`. PART4 §15 marked.

- **SESSION ORDER item 9 — the last Part 1 backlog picks** (§D) — CLOSED 2026-10-28 (PLAN_W:
  `35e827e` the fill checks require a 0..n−1 frame, `aff21f8` the POI record gate `active OR locked`, `321667a` a sub
  probe's reference = its derived CTS zone as of the read, `538cd2f` the sub CTS KL zones exported + drawn hidden
  behind the M15 `Sub CTS zones` button; saves S_W9b `20261028_165600_aff21f8`, S_W9c `20261028_180228_538cd2f`).
  Items 1–8 and their plans (PLAN_H … PLAN_V): §D "SESSION ORDER" + memory `project_session_order_status.md`; + the
  KL hover's `kind=` line (PLAN_W §4, S_W9d). Week 9 Part 1 itself stays OPEN for item 10 (the parked-items sweep,
  the user 2026-10-28). Not pushed (the user's call).
- **The cycle-0 carve-out revisit** (§D 5b; raised by the user 2026-10-26, scheduled 2026-10-27) — CLOSED 2026-10-27
  by PLAN_T in two landings: T1 `453f227` (the first_confluence probe's MS reads what its sub reads; zero change; save
  S_T1 `20261027_131836_453f227`) and T2 (the carve-out kept on the main sid 0 + the sub its cycle-0 first_confluence
  creates only; subs 4, 7, 8 confirm cycle 0 by proximity — `017c920`, save S_T2 `20261027_140757_017c920`). Every
  round verbatim:
  `engine_v2/plans/qa_cycle0_carveout_inputs/user_spec_20261027_cycle0_carveout.md`.
- **WVMI back on the chart** (§G; deferred 2026-05-28 `e64556a`, un-tabled by the user 2026-10-24) — CLOSED 2026-10-26
  in five landings (`257f8ba`, `76b5a17`, `f63df34`, `588014a`, `2dc9e5d`; save S_wvmi_marks
  `20261026_230619_2dc9e5d`): no H1 gate, the furthest-close LP rule, the per-LP rows + `*_wvmi_lps.csv`, and the
  WVMI LP marks above the volume bars on the three charts (CHARTING_SPEC §13) — not the old wave-candle hover, which
  stays without WVMI. The user's request, every answer verbatim and each step:
  `engine_v2/plans/qa_chart_edits_inputs/user_spec_20261024_wvmi_on_chart.md`.
- **Naming-Standard audit of never-audited index keys** (§A) — CLOSED 2026-10-12 by PLAN_N
  (`engine_v2/plans/PLAN_N_naming_audit.md`; committed `1b2f24c`, saved S_N `20261012_165925_1b2f24c`): KL steps
  `base_idx` / `expanded_idx`; RANGE_STARTED `range_anchor_idx` (MS's range start on every path) / `confirmed_at`;
  `effective_idx` == `ev.idx` always; "collapsed parent cycle". Charts byte-identical on the window.
- **H1 range hover lines snapshot every structure** (§A) — CLOSED 2026-10-12, committed `9f9406d`, saved S_N_hover
  `20261012_194022_9f9406d` (save `cc3d3ff`, trunk `efd95b9`): widened by the user to option B — the hover line = the
  box, candle for candle (snapshot at each event's replay candle); it also fixed a second, same-structure symptom (a
  label range's confirm candle showed its anchor-time state: 16 candles). `/compare` vs S_N = the prediction (4 H1
  traces, 28 candle points); the user "looks good go ahead". Detail: `engine_v2/plans/plan_n_inputs/
  followup_hover_per_structure.md` "The landing decision".
- **Dead code in the H1 range hover-line builder** (§A hygiene) — CLOSED 2026-10-12 in `9f9406d` (deleted with the
  hover fix, the user: in the same change; no output effect).
- **§D.1 item 2 — the cross-limit price debugging (PLAN_M) + its follow-up CL-8** — PLAN_M COMPLETE 2026-10-12 (S_M2
  `20261012_023137_10e4c6f`); CL-8 (sub 3 cycle 2's IC 2821 vs H1 sid 1 cycle 1's 732) CLOSED 2026-10-12 with no change
  — the user: "1, keep it — close CL-8 with no change" (PLAN_M §17). Item 2 has no open work left.
- **§D.1 item 2 — the cross-cycle fib revamp (one logic for main and sub)** — COMPLETE 2026-10-09: PLAN_I S1–S6 +
  PLAN_J (J1 / J2 / J3 + J3's swap → a raise); saved S_S6 `20261009_020650_5f6bae3`; charts reviewed by the user
  ("they look good"). The record stays in §D.1 item 2; canonical spec `engine_v2/zones/CROSS_CYCLE_FIB_SPEC.md`.
- **Cascade-viz hover extras** (G: `current_versions` vs peak; a V90 → V60 → V30 downgrade cue) — CLOSED 2026-10-06 as
  obsolete: PLAN_H landing B removed the variants (and `versions` / `current_versions`) from the POIs; the POI hover
  shows the institutional pattern instead (HV). Detail: memory `project_item_5_cascade_viz.md` (historical).
- **`PRE_REFACTOR_INVARIANTS.md` deleted** — 2026-09-30 (PART4 §13 item 7): every item was already canonical or
  superseded (UC1, the three-probe family, the May checkpoints, `meta["active"]`, the 9/6/3 pips) except the
  `cts_phase` values → `MARKET_STRUCTURE_SPEC.md` "State machine overview" (`FALSE_BREAK` declared, never assigned).
  Memory `project_part4_workflow.md` retired; its handoff-prompt practice kept (memory
  `feedback_session_handoff_prompt.md`).
- **§13.7 doc finalization** — done 2026-09-30, IN PLACE (user): PART4 stays the canonical multi-TF spec (status
  header, §13 step statuses, footer); the split moved to G. The §16.10 file-naming deviation resolved as SPEC FOLLOWS
  CODE (`{basename}` + `_{TF}_{lens}` suffix, one flat folder per save — §12 / §16.10 / §16.11 rewritten), with a
  nesting rule for later (every path segment after `H1.main`: `_M15_counter__M5_confluence`).
- **§13.5.e remainder** — CLOSED 2026-09-30 WITHOUT deleting the `s_res.df.attrs[...]` block (user decision): measured,
  the registry's `H1.main` entity IS `s_res.df` and an entity's artifacts live in its df's `attrs` (PART4 §9.2 / §9.3),
  so the block is the registry's store and both charts already read it through the registry. Landed: the stale
  "DEPRECATED" comment relabelled, 2 dead `run_replay.py` writes deleted, PART4 §9.3 / §13.5.e + LANDMINES fixed;
  byte-identical (24 CSVs + figure JSON). PART4 §13.5.e "CLOSED" note.
- **Week 8 hygiene pass** — 2026-09-30, all byte-identical (24 CSVs + figure JSON; only the `a8ed1cc` wave-candle
  hovers differ from the save): the dead imbalance helpers `get_unfilled_imbalances` / `has_imbalance_in_range`
  (`f630640`); the legacy MS functions `compute_structure_scenario_3` / `compute_structure_from_start` /
  `identify_start_scenario_2_after_reversal` + their private helpers + `tests/test_scenario3.py` (`1071347`); the M15
  chart's inert fib toggle + never-applied dot opacity tier — charting-audit #1 / #2 (`cd8eb9e`); the stale-docs sweep
  (proximity 8/5/3 + probe-reset 4/3/2 pip values, Commit 2 wired, (c) done, the Week-6 overview, the §13.5.d
  docstrings, charting-audit #10 + the verified doc-only items) (`45912ac`) + the stale memory notes; `stash@{0}`
  dropped (`6fdb993`). User decisions: register "Week 8 close-out decisions" commit `ee2314e`.
- **Syllabus Week 8 extras** (pair / TF dropdowns, a "show HTF context" checkbox, `ContextSnapshot`) — DROPPED by the
  user 2026-09-30 as superseded by Part 4: one static chart per entity / lens replaces the dropdowns + checkbox, and the
  entity registry holds every snapshot field (HTF trend = the H1.main structure; active HTF zones = the H1 zone attrs;
  LTF candidates = the `TriggerRecord`s / subs). `engine_v2/SYLLABUS.md` Week 8 notes it.
- **"(b) the (0,0,1) start-logic update"** — closed 2026-09-30 on the memory's own evidence: it IS the true-first-
  breakout cycle-0 redesign (design locked 2026-06-07, Commit 1 2026-06-08, Commit 2 2026-06-16) — memory
  `project_unified_identify_start_probe.md` ("the (b) start-logic update is a FULL DESIGN … see
  project-true-first-breakout-cycle0"), `project_true_first_breakout_cycle0.md`.
- **`get_unfilled_imbalances` direction filter** (was H) — dropped 2026-09-30 with the user's "delete" decision (A).
- **Zone-proximity candle-direction filter** — re-discussed 2026-09-30: a no-op on the reference window (all 6 H1
  triggers + both sd-proximity CTS confirmations already point into their zone; in May it cut 48 → 26, before Rules
  1-3); user: no trigger filter — triggers stay structural; the idea moves to D (entries). Filter + revert: `47e7532` /
  `2afd3b2`.
- **POI two-stroke review over the full window** — done 2026-09-30: every POI's on/off history re-derived per candle
  and consistent (`review_scripts/poi_cond3_check.py`); only 2 POIs ever deactivate (H1 s1c2, sub 7 s0c0); the charts
  already draw each active stretch; user: OK, no change. Memory `project_item_3_poi_lifecycle.md`.
- **Half-clipped reversal check** — done 2026-09-30: 0 on the reference window (every kept `REVERSAL_CANDIDATE`
  applies at its own sub's window end); the limit itself stays — bucket E. PART4 §17.12.
- **Wave-candle hover "BOS zone:" on CTS lines** — fixed 2026-09-30 `a8ed1cc`: kind + role label from one builder
  (`wave_candles.wave_candle_hover_lines`); 55 of 110 labels had been wrong. `engine_v2/zones/WAVE_CANDLES_SPEC.md`
  "Rendering simplifications".
- **WVMI pass** — Plan G landed 2026-09-30 (`83c7a12`, fold-in `576bf6f`, Int64 `3781cab`); `engine_v2/plans/PLAN_G_wvmi_unique_sub.md` §9.
- **§13.5.d paired carve-out removal** — subsumed by the sub-structure lifecycle redesign (2026-05-25); leftovers in A.
- **KL lifecycle convention** (`meta["active"]` / `deactivated_by` indirection) — gone from the code (Phase 3 +
  the pool).
- **Cross-cycle fib unification arc** — user sign-off 2026-06-19 (only the bundled follow-up in E remains).
- **Main sid=0 cycle=0 true-first-breakout (Commit 2)** and the **(c) zone-pattern inversion pass** — done
  2026-06-16 / 2026-06-20.
- **Never-established-cycle fallback POI** — resolved 2026-09-29 (option N; PLAN_D §7.3).

---

## Idea Template (for a new idea)

### Idea Name
**Problem:**  
**Proposed Solution:**  
**Integration Point:**  
**Why Deferred / Trigger:**  
Then add its one-line entry to the right bucket above.
