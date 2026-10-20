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
- **Charting audit 2026-09-21 — the style / code items** — #7 sub-native zone outline hardcoded (no `zone.m15.poi.*`
  key), #8 hover-label colours hardcoded, #9 the M15 zone-hover hitbox lines hardcoded (H1 reads the registry), #11 the
  unused `structure.reversal_watch_line` key + the never-plotted `watch_y` series; plus the audit's still-unverified
  list. Also: 8 unused locals in `charting/export_m15_chart.py` (ruff F841, pre-existing). OPEN (fold in when those
  files are next touched). Detail: memory `project_charting_audit_20260921.md`.
- **Two dead reads found by the Week 9 /prepare + item-1 talk** — `_run_multi_tf_dual(poi_zones=...)` is never read
  (`pipeline/orchestrator.py` ~:837; fold into item 1's landing B, which touches the orchestrator's `POIConfig`);
  `wvmi._find_temporary_lp` computes `fp_vol_dir` and never reads it (fold into "The agreed direction for wave candles
  + WVMI", D.2). OPEN.
- **The H1-main frame is classified and stamped "H1" whatever the config timeframe** — `orchestrator.run_pipeline`
  calls `apply_candle_classification(df)` with its default (`timeframe="H1"`), so a re-enabled M15-main window
  (`config.py` keeps them commented out) would get H1's pinbar floor AND the institutional pattern's H1 cap / params
  through the frame's `attrs["timeframe"]` (PLAN_H landing review, 2026-10-06). Pass `CONFIG.timeframe` (or assert H1)
  when such a window comes back. OPEN; not reachable on the current config.

## B. Week 8 scope — decided 2026-09-30 (the M5 layer moved to D; the syllabus extras dropped — "Recently closed")

## C. Part 4 closure — DONE 2026-09-30 (all three items in "Recently closed"; Part 4 is complete)

## D. Week 9 Part 1 — Entry prerequisites (before Entries)

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
3. **Zone proximity triggers — changes and edits to the opposite-direction (CTS-zone) trigger** (the user brings the
   details). Today: `engine_v2/zones/zone_proximity.py` — per (sid, cycle) with a `CTS_CONFIRMED`, triggers alternate
   sd (BOS / POI zone) → opp_sd (the CTS zone's inner bound) → sd …, at H1 8 / M15 5 / M5 3 pips; narrow cycles: Rules
   1–3 (LANDMINES "Narrow-Cycle Rules 1+2+3 Are a Triple"). The opp_sd (CTS-prox, "var 3") entry is what
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
   folded in; build order 0a → 0b → 1 → 2, 4 commits, `/commit-save` only after step 2. **BUILD UNDER WAY
   2026-10-20** (PLAN_R §8 = the build log): steps 0a `a6e71c0`, 0b `c1bd05a` and 1 (the ladders + the narrow proximity-reference cap on parent zones) LANDED, all zero change as predicted; step 2 (the overlay in the sweep) next.
4. **Zone size adjustments** — D.1 item 3 below (the user's spec).

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
   the ranking rules, L2; Step 4 — D.2 "Step 4 of IC identification", deferred). Still open from the debugging log: sub 3 cyc 3
   has no POI record (PLAN_H §8.9). **The write-up RECEIVED 2026-10-09** (verbatim
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
3. **Zone size adjustments** — **4th in the SESSION ORDER above (user 2026-10-12: after the hover follow-up, the IC /
   cross-fib questions and the opp_sd proximity-trigger edits).** User 2026-09-30:
   "currently some KL & POI zones may be very large or small in their
   current base form. This will cover additional logic that adjust the size." The user provides a detailed spec.
   The Naming-Standard audit landed first (PLAN_N, 2026-10-12: a KL step's index is `base_idx` on INIT, `expanded_idx`
   on an expansion — read via `kl_zones_v1.bounds_step_draw_idx`). Related: "Zone strength scoring" (D.2). Detail: `engine_v2/zones/KL_ZONES_SPEC.md`,
   `POI_ZONES_SPEC.md`; measured sizes: memory `project_week9_part1_inputs.md`.
4. **Pinbar body-pip floor calibration** — scheduled here by the user 2026-10-01 (was TRIGGER: strategy optimization):
   the provisional values (H1 2.2 / M15 1.0 / M5 0.5 pips, NZD_USD only) and the reclassify-to-pinbar approach itself.
   Iterative: parameter flips + chart review; a candle-type change cascades (patterns → structure → zones), so every
   flip gets a full `/compare`. Detail: memory `project_pinbar_body_pip_floor.md`,
   `engine_v2/features/CANDLE_BODY_FLOOR_NOTES.md`.

**D.2 — Backlog: decided after D.1 is complete (user 2026-10-01).**
- **Entry confirmation candle patterns** — user 2026-09-30: "define new candle patterns that will be used to confirm
  entry (when necessary). This is a prerequisite for Entry, but can go last in Part 1 or first in Part 2." Related:
  "Candle-direction / rejection confirmation at a zone touch" below; D.1 item 4 (the pinbar floor); the pattern engine
  (`patterns/structure_patterns.py`, `features/candles_v2.py`). OPEN.
- **The M5 layer** (`H1.main >> M5.counter` — a third lens directly under H1.main, the Part 4 vision's 4th chart;
  memory `project_part4_vision.md`) — DECIDED 2026-09-30: its own Heavy block (plan + cold review), before any Entries
  work that reads M5; not a Week 8 blocker. Footprint seen so far: `debug/export_sub_tables.py` hard-codes `_M15_`, and
  `pipeline/orchestrator.py` (~:646) runs the whole multi-TF path only when `"M15"` is in `lower_timeframes`. Related
  (only if M5 ever nests under an M15 sub): the canonical encoding of a deeper path, e.g. `H1.main >> M15.counter >>
  M5.confluence` (PART4 ~:176 "Open (deferred)"; PART4 §13.6, §17.12). BACKLOG.
- **Zone strength scoring** — syllabus Week 7 (`strength_score` / `strength_flags`, "strong zones look strong", click a
  zone → strength + flags); `KLZone.strength` is always `0.0` (`zones/kl_zones_v1.py`). DECIDE (scope). Idea attached:
  scale strength thresholds by ATR percentile (volatility regimes).
- **Candle-direction / rejection confirmation at a zone touch — as an ENTRY condition** (user 2026-09-30: not a trigger
  filter; triggers stay structural). The May trigger filter (`47e7532`, reverted `2afd3b2`) is the reference form:
  trigger candle body opposing the zone side. DECIDE when Entries are designed. Detail: this register's "Recently
  closed" (the 2026-09-30 measurement).
- **A same-candle replacement chain names a replacer that never starts** (PLAN_I S1's landing review, 2026-10-07;
  latent since Plan C, 0 on the window): when two records start on the same candle and the later one replaces both
  the incumbent and its co-starter, the sweep's phase 3 takes the FIRST equal-priority end entry, so the incumbent's
  `ended_by` names the co-starter that was itself replaced — a sub that never starts. Before S1 the triggers CSV
  carried that sub's creation id (a sub no chart draws); since S1 the cell is empty and the sweep logs `WARNING
  [sweep] record … ended by pool#k, which never started`. Candidate fix: `ended_by` = the replacer that survives.
  Pinned: `test_lifecycle_sweep_unit.py::test_a_replacer_that_never_starts_leaves_ended_by_empty_and_warns`. DECIDE.
- **Should subs get the DERIVED CTS KL zone as their probe reference?** (today every sub reference zone is ad-hoc).
  Unscheduled follow-up. DECIDE. Detail: PART4 §17.12, GOTCHAS (the `kl_zones=[]` pool-path note).
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
  (sub 1 c0, c3); the user (2026-10-13): "For now it is" (no POI). DECIDE (whether to build it at all).
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
- **The agreed direction for wave candles + WVMI** (PLAN_H Q-W) — item 1 gives the IP legs the *agreed direction*
  (candle and volume direction agree; one at 0 → the other decides; both 0 = a true doji). Wave candles and WVMI keep
  GLOSSARY "qualified candle" (no volume-only-doji row) for now — user 2026-10-05: "we likely will also make the change
  for wave candles & WVMI, but we'll come back to that later". "Qualified" is implemented twice
  (`wave_candles._is_qualified`, inline in `wvmi._find_temporary_lp`) → one helper first; outputs change → its own
  `/compare`. DECIDE (when).

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
  later"); the MS vs FibTracker activation divergence (measure first); latent: `inst.meta` mutated on imbalance
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
- **Opposite-direction POIs** — the zone-proximity logic assumes "POI ⟹ sd"; TRIGGER: any feature producing an opp_sd
  POI → revisit `check_zone_proximity`. Detail: `engine_v2/zones/POI_ZONES_SPEC.md` (~:420).
- **The start-candle probe's market structure runs at the H1 proximity values on M15** (8 pips and the 50-pip
  narrow-cycle gap instead of 5 / 30): `unified_probe.py`'s Phase 2 builds it through
  `structure_engine._make_market_structure` without `timeframe` (that helper's docstring: "a deferred cleanup").
  INERT BY DESIGN — the probe's result reads two CTS_ESTABLISHED, cycle 0's CTS_CONFIRMED and a reversal, never a
  proximity confirmation, and cycle 0 cannot be proximity-confirmed (`unified_probe.py:549-552`, PLAN_I §9.2 F7).
  TRIGGER: the probe ever reads a proximity-confirmed CTS → pass the sub's `timeframe`. Registered 2026-10-19 (the
  user: "We can add it"); detail `engine_v2/plans/qa_opp_sd_zones_inputs/qa_findings.md` rounds 5–6.

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
- **Uniform 3-tier chart opacity** (brightest = cycle not ended, mid = sid not ended, faded = sid ended; one
  `opacity_for(sid, cycle)` helper for every element) — its own session. Detail: FIB_LIFECYCLE_SPEC §9.4 + §13.
- **Breadcrumb header with cross-chart navigation** — "not in v1". Detail: PART4 §16.9.
- **Split `PART4_REFACTOR_SPEC.md` into canonical spec files** (a new `MULTI_TF_SPEC.md` + sections moved into
  `MARKET_STRUCTURE_SPEC` / `CHARTING_SPEC` / `LANDMINES`) — the drafted §13.7 plan; PARKED as optional by the user
  2026-09-30 (§13.7 was finalized in place instead: ~235 `PART4 §x` citations, no behaviour gained). Detail: PART4
  header "Canonical" + footer.
- **Would-have-been subs for collapsed parent cycles** (the user, 2026-10-19: "OK we can go with parent zones only
  for now and consider subs later"): the opp_sd zones proximity overlay (SESSION ORDER item 3, the coming PLAN_R)
  reads sub zones only on parent cycles with a real-time lifecycle; a collapsed cycle (lifecycle floor ≥ end — on
  the reference window structure 1's cycles 0 / 1, `[902, 902)`) scans its parent zones only, because the pool
  builds no subs there (LANDMINES "Collapsed Parent Cycles"), and a live sub of the real-time parent cycle is NOT
  borrowed (decided). The engine's would-have-been clock is the event MOMENT (`ev.idx`); subs in that clock would
  need probes / records outside the real-time pool. Detail: `engine_v2/plans/qa_opp_sd_zones_inputs/qa_findings.md`
  rounds 4–6.

## H. Deferred indefinitely

- `armed_idx` chart surfacing; consumer-specific fill predicates (IMBALANCE_FILL_SEMANTICS "When to revisit").

## I. Needs confirmation (possibly superseded)

- (none — the last item was closed 2026-09-30; see "Recently closed")

---

## Recently closed (history pointers)

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
