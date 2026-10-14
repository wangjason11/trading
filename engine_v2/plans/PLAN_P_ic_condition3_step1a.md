# PLAN_P — IC condition 3 = the immediate unfilled imbalance + Step 1A, the 80–100 fallback zone (the user's changes 1 + 3, 2026-10-13)

**Status: LANDED 2026-10-13** — source `23f395d`, save **S_P** `20261013_231327_23f395d` (save commit `69ac1d7`, trunk
`ab4c982`; = `LATEST_week9-entries-decisions`; the save replay md5- / figure-hash-equal to the variant again) (§10: `/compare` = §6 exactly, mutants 27 / 27; landing review 384,641 tokens, no
BLOCKER / MAJOR; the user's chart look: "Charts look good, go ahead and commit once the review is done"). Plan review: 2 lenses, 600,960 tokens (§9). Baseline **S_O** `artifacts/commits/week9-entries-decisions/20261013_151425_ce5ac9b`
(= `LATEST_week9-entries-decisions`; source `ce5ac9b`, PLAN_O). HEAD at drafting `d8f6279`; suite 1718. Session order
item 2 (register `IDEA_PARKING_LOT.md` §D "SESSION ORDER"). ONE commit — the revert unit (the user, 2026-10-13: "change
1&3 are something we may choose to revert or update in the future, so having them as 1 commit together makes it easier
to revert to down the line").

Inputs (`qa_ic_crossfib_inputs/`): the user's write-up verbatim `user_writeup_20261013.md`; the Q&A, the measurements
and every decision verbatim `qa_findings.md`; the starting code `trial_changes_1_3_on_S_O.patch` (the combined trial
minus change 2); the variant replay's fingerprints `plan_p_variant/` (§6).

---

## 1. The rules

**Change 1 — IC condition 3 (the user's write-up, verbatim):** "Instead of the current requirement 3 under Step 1, what
I want to do instead is take one of our tiebreak / general ranking rules and replace the current requirement 3.
Specifically the rule: Prefer IC immediately followed by unfilled imbalance, within IC + 2 (<=) … Thus, before even
applying IP search, I want to keep the candidate list to ONLY candles where we have immediate imbalance after rather
than just any generic imbalance anytime after the IC." And: "Since the immediate unfilled imbalance rule moves up to
Step 1, they are no longer needed in Steps 2 & 3, which shorten those rulesets."

**Change 3 — Step 1A (verbatim):** "Step 1A will be run only if Step 1 returns no candidates. In that case, then for
Step 1 requirement 4, we change it to: The candle must overlap with the 80 - 100 Fib zone … From then on, we apply Steps
2-3 but now effectively to only candidates from the 80-100 zone … it would only ever come into play immediate after
Step1."

**Decided (2026-10-13, verbatim in `qa_findings.md`):**
1. With change 1 the last rule of BOTH rule sets picks the OLDEST candidate in play (its own immediate gap is the
   earliest after it; Step 3: or the neighbour sharing that gap) — "Yes".
2. A Step 1A IC's POI activation re-checks the touch in 80–100 — "Yes, POI activation check should always reference the
   zone parameters the IC came from".
3. Records left with no IC (all-pinbar in 61.8–80 — Step 1A does not run; nothing in either zone) — "For now it is."
4. **The POI activation's imbalance condition STAYS as today, decoupled from IC identification:** "The concept of
   keeping a POI zone active centers more around that we currently still have unfilled imbalances in our current cycle.
   Therefore, keeping the old rule suffices in checking that. IC identification on the other hand is trying to find the
   best candidate to build the zone around, making the immediate imbalance part crucial."
5. **The old condition 3** (`has_unfilled_imbalance_after`, its IC CSV column): **"Keep as information"** (this
   session; offered: drop — recommended / keep as information / keep renamed).
6. **Names: "Keep the trial names"** (this session): `ic_zone_band` = `"61.8-80"` / `"80-100"`, `IC_ZONE_PRIMARY` /
   `IC_ZONE_FALLBACK`, `POIConfig.ic_fallback_fib_min` / `ic_fallback_fib_max` (80 / 100), the POI hover line `IC zone
   80-100 (fallback)`, the fib status suffix ` · zone 80-100 (fallback)`, the spec's "Step 1A — the 80–100 fallback
   zone".
7. **The BOS anchor candle as a Step 1A IC** (plan review lens A #3: the 100 % level is the BOS anchor's extreme, so
   that candle always touches 80–100; confluence sub 1 cycle 2's Step 1A IC 1934 IS its BOS anchor, beside the pinbar
   1935): **"Keep it"** (this session; offered: keep (recommended) / exclude from Step 1A / exclude from both steps).

---

## 2. The engineering reading

**Step 1** (`check_ic_conditions`): one check per agreed-−sd candle of the fib's BOS → CTS window — condition 2 (the
IC reference, unchanged), **condition 3 = `immediate_unfilled_imbalance_first_c2_idx` is not None** (an unfilled sd
imbalance whose first gap candle is at IC + 1 … min(IC + `immediate_imbalance_candles` (2), the CTS anchor); the
two-stroke fill through the CTS anchor, `evaluated_at=None` — the window and the test are the removed rule's, PLAN_H §6
TBU-1 / TBU-2 / TBU-5), condition 4 = the touch of 61.8–80. The immediate gap is now computed on EVERY check, in both
consumers (it was computed on classified checks only — the drawn path classifies all, MS's read only the candidates).
`has_unfilled_imbalance_after` stays computed on every check as information (decision 5) — the pick never reads it.

**Step 1A** (`select_ic`): when Step 1 has checks but no IC candidate, condition 4 is re-evaluated on the 80–100 zone
(`ic_zone_config(config, IC_ZONE_FALLBACK)` → `fib_zone_bounds`), conditions 1–3 reused (none depends on the zone);
`ic_zone_band` records which. Then Steps 2 → 3 as usual on those candidates. Never after Steps 2–3: an all-pinbar
61.8–80 answer stays `all_ic_candidates_pinbar` (band 61.8-80). A record whose Step 1A finds none: `no_ic_candidate`
with band 80-100 (or `all_ic_candidates_pinbar` with band 80-100 — every 80–100 candidate a pinbar). No checks at all
(a fib neither active nor locked, or no agreed −sd candle) → band 61.8-80 (Step 1A does not run: `checks` empty).

**Steps 2 / 3 lose the immediate rule.** With change 1 every IC candidate passes it, so it is a no-op: the tiebreak
filter keeps every tied IC (≥ 2 → never exactly one), the ranking filter keeps every in-play candidate (≥ 2 after rule
1). Removing it changes no output — proven by the variant replay (§6: the trial kept the rule; the landing must be
byte-identical to it). The tiebreak rules become (1) the IC's candle type, (2) the most recent before the imbalance;
the ranking rules (1) exclude pinbars, (2) maru > normal, (3) the most recent before the imbalance. The values
`immediate_unfilled_imbalance` of `ip_tiebreak` / `ranking_rule` disappear (a save before S_P can hold them). The
last rule's reference = the oldest candidate in play's OWN immediate gap — `most_recent_before_imbalance` takes the
minimum start of the unfilled gaps after it; consecutive sd gap candles merge, so at most one sd instance overlaps a
2-candle window; none straddles a checked candle (agreed −sd ⇒ candle direction −sd or 0); both reads use the CTS
anchor as the fill horizon — so the pick = that oldest candidate (Step 2: always, TBU-4) or its in-play neighbour
sharing the gap (Step 3, ICR-1) — decision 1.

**80–100 vs 61.8–100 (plan review, both lenses):** on the PICK the two readings of Step 1A are indistinguishable —
Step 1A runs only when no check passes conditions 2 + 3 + the 61.8–80 touch, so a check touching 61.8–80 already
failed 2 or 3. The band shows only in (a) the CSV's `touches_zone` of a Step 1A record and (b) the activation's
touch re-check: before the CTS extends, a Step 1A IC can sit in the as-of 61.8–80 zone and not touch the as-of
80–100 one → inactive (`ic_not_touching_fib_zone`) — by decision 2; 0 such deactivations on the window (the 4 Step
1A POIs). The BOS anchor candle always touches 80–100 (decision 7). The fib box and its hover's 61.8 / 80 prices stay
61.8–80 on a Step 1A record — a fallback POI is drawn outside the fib box.

**The activation sweep** (`_compute_poi_activation_history`): condition 3 unchanged in code — ANY sd imbalance formed
by t overlapping `(ic_idx, t]`, not commit-filled (decision 4: reworded as the POI's OWN condition, not "IC condition
3 re-checked"); condition 5's touch in the IC's band (`config=ic_zone_config(config, rec.selection.ic_zone_band)`,
decision 2).

**MS's in-flight read** (`poi_inner_for_fib`) calls the same `select_ic` (PLAN_H D2) — it gets condition 3 and Step
1A with no code change, so MS's sd-proximity CTS confirmation can read a Step 1A IC's inner (by D2's design: one IC
per fib for the drawn POI and MS). Measured (`qa_ic_crossfib_inputs/shadow_plan_p_ms_reads.py` →
`plan_p_variant/shadow_ms_reads.txt`, plan review lens A #4): H1 — 0 of 56 reads take a Step 1A IC (Step 1A ran in
35, found none); M15 — 43 of 429 reads (7 distinct fib-version / IC pairs: the as-of fibs of sub 0 c0 / c1, sub 1 c2,
sub 3's cross 1→2 (IC 2808 as of its CTS then), sub 7's cross 0→2 (4131), in slice-local indices); none moves an M15
event (the only M15 event-meta change, 1902, reads IC 1890 — band 61.8-80). The H1 cascade (§6) is change 1's.

**Not changed:** the IC reference (PLAN_O), the pattern classifier, the stamp (`stamp_ic_references`), the
lifecycle gate, the activation floor, the zone-proximity scan (it reads the drawn POIs — fewer of them, §6), the
unwired V30 / V60 / V90 path (`find_ic_variant_candidates` keeps `_has_unfilled_imbalance_after` as its filter).

**The debug CSV:** `touches_zone` = the touch of the record's `ic_zone_band` (80–100 on every row when Step 1A ran —
the 61.8–80 touch is not exported then; the band column says which). New column `ic_zone_band` after
`is_ic_candidate`, every row incl. the no-IC row.

---

## 3. Every reader

| Reader | Where | What changes |
|---|---|---|
| Step 1 / Step 1A / Steps 2–3 | `select_ic`, `check_ic_conditions`, `ICCandleCheck.is_ic_candidate`, `apply_tiebreak_rules`, `apply_ranking_rules` (`zones/poi_zones.py`) | condition 3; Step 1A; the immediate rule removed |
| The drawn POI | `select_ic_records` → `derive_poi_zones` | the picks; meta `ic_zone_band`; the activation touch in the band |
| The IC CSV | `derive_ic_candidates` → `debug/export_zones.export_ic_candidates` | column `ic_zone_band`; `touches_zone` in the band |
| MS's in-flight POI inner | `poi_inner_for_fib` | the same picks (no code change) |
| The POI hover (H1 chart, the M15 lens charts' sub POIs + their H1 overlay) | `_zone_render.poi_ic_hover_text` | `<br>IC zone 80-100 (fallback)`; the immediate tiebreak line removed |
| The fib hover's status cell (both charts) | `_zone_render.fib_hover_status_text` | ` · zone 80-100 (fallback)` |
| The ranking hover | `_zone_render._poi_ranking_hover_text` (`_RANKING_HOVER_PARTS`) | the immediate entry removed |

Passive: the sub mirror (`entity_df_mutation`) — `ic_zone_band` is a `str` (never shifted; `dataclasses.replace`
keeps it on the IC records; the POI meta dict is copied); the zone-proximity scan + the WVMI gate read the drawn POIs.

---

## 4. The code (uncommitted on `d8f6279`; = the trial patch + the cleanup)

1. `zones/poi_zones.py` — the trial's change 1 + Step 1A + `ic_zone_band` (POIConfig, `IC_ZONE_*`, `ic_zone_config`,
   `ICSelection.ic_zone_band`, the POI meta, `ICCandidateRecord.ic_zone_band`, the activation's `ic_zone_config`); then:
   - removed: `TIEBREAK_IMMEDIATE_UNFILLED_IMBALANCE`, `RANKING_IMMEDIATE_UNFILLED_IMBALANCE`,
     `filter_immediate_unfilled_imbalance`, `apply_tiebreak_rules`'s `immediate_c2` parameter + rule 1,
     `apply_ranking_rules`' rule 2, `select_institutional_pattern`'s immediate dict;
   - `ICCandleCheck.immediate_imbalance_first_c2_idx` REQUIRED (no default — condition 3 is always computed; moved
     before `pattern`); `ICSelection.ic_zone_band` REQUIRED (every construction passes it; asserted in the two bands,
     and the fallback band only with checks — review A #10 / B #9);
   - docstrings / comments: the module docstring, `_has_unfilled_imbalance_after` (retired, information + the V30
     finder), `POIConfig.immediate_imbalance_candles` (condition 3's window), `fib_zone_bounds`,
     `immediate_unfilled_imbalance_first_c2_idx` (condition 3; asserted for every check), the constants' comment,
     `ICCandleCheck`, `check_ic_conditions`, `most_recent_before_imbalance` (decision 1), both rule sets (renumbered),
     `ICSelection`, `select_ic` (Step 1A's scope), the activation comment block + docstring (decision 4, the band),
     `ICCandidateRecord`. Every TRIAL wording → PLAN_P.
2. `charting/_zone_render.py` — the trial's two hover additions; the immediate tiebreak branch + the ranking entry +
   the two imports removed; `meta["ic_zone_band"]` / `ic_row.ic_zone_band` direct (no `.get` / `getattr`) AND
   asserted in the two bands in both builders (review B #4: `ICCandidateRecord.ic_zone_band` defaults to None, so the
   direct read alone did not fail loudly on a row built without it); docstrings.
3. `debug/export_zones.py` — the column after `is_ic_candidate`; the docstring.
4. Comments outside the patch (review B #3): `multitf/entity_df_mutation.py` (`_ZONE_META_IDX_KEYS`' tiebreak /
   ranking comments), `patterns/institutional_pattern.py` (module docstring: the touch incl. Step 1A).

---

## 5. Docs (same commit)

- `zones/POI_ZONES_SPEC.md` §3 — the steps list (Step 1 condition 3, Step 1A, the shortened rule sets), §3.1 (condition
  3 = the immediate gap; the retired one as information; condition 4 + Step 1A), §3.4 (the tiebreak rules: 2 rules; a
  dated note for the removed one; the TBU-4 sentence = always), §3.5 (the CSV: `ic_zone_band`, `touches_zone` in the
  band, `immediate_imbalance_first_c2_idx` = condition 3, `has_unfilled_imbalance_after` retired), §3.6 (the ranking
  rules: 3 rules; the H1 sid 0 cycle 0 example rewritten — it argued the order vs the immediate rule; a dated note),
  §4 "Zone Data Fields" (`ic_zone_band`), §4 "Lifecycle" (cond3 = the POI's own imbalance condition, decoupled — the
  user's words; cond5 in the IC's band), "Data Flow" (`:893`), each with this landing's values where the section
  quotes the window.
- Plan-review additions (both lenses): POI_ZONES_SPEC Overview (`:11-12`), §1 (`:60-62`, `:94-96`), the §3.2
  window paragraphs (+ a PLAN_P line), §3.5's `ip_tb_*` / `ip_tied_count` descriptions, §4's value lists, Zone
  States + Lifecycle "IC qualifies", **§5 "Hover Data"** (the tiebreak line, the 2575 example, the fallback line +
  suffix), Implementation Status; `ARCHITECTURE.md` (`:290-305` the activation conditions, `:323-325`);
  `IMBALANCE_FILL_SEMANTICS.md` `:87`, `:222`, `:224`, `:267` (+ the listed); `LANDMINES.md` `:2399`; `GOTCHAS.md`
  `:2062-2063` (the expected 860 — PLAN_P picks 859, §6); CHARTING_SPEC `:167` / `:170` (the fib box stays 61.8–80);
  superseded pointers also on `plans/plan_h_inputs/ic_identification_revamp_spec.md` and PLAN_L §1 / §3.1; stale
  tools also `plan_h_inputs/cmp_l2.py`, `ic_general_ranking_shadow.py`.
- `charting/CHARTING_SPEC.md` (~`:148`, the POI hover's tiebreak line + the fallback line; the fib status suffix).
- `GLOSSARY.md` — `*_ic_candidates.csv` (`:178`), "tiebreak rules" (`:179`), "ranking rules" (`:180`); a new entry
  "Step 1A / `ic_zone_band`"; "IC candidate" if it names condition 3.
- `IMBALANCE_FILL_SEMANTICS.md` (`:196-198`, `:221`, `:223`, `:289`) — condition 3 = the immediate gap; the old test =
  information + the V30 finder.
- `LANDMINES.md` (`:1755`), `structure/MARKET_STRUCTURE_SPEC.md` (`:447-448` — MS's imbalance reads), `GOTCHAS.md`
  (`:2059` — the tiebreakers' list; `:2051` historical, a dated pointer).
- Superseded pointers (text kept): `plans/PLAN_H_institutional_pattern_poi.md` §6 TBU-1 … TBU-5 + ICR-9 (the immediate
  rule's helper) — "the immediate rule became IC condition 3, PLAN_P"; `plans/PLAN_L_ic_identification_steps.md` §4.1
  (ranking rule 2); `plans/plan_h_inputs/ip_tiebreaker_update_spec.md` (one header line).
- Stale-tool headers (one line, like PLAN_O's): `plans/plan_h_inputs/mutants_tbu.py`, `mutants_l1.py`, `mutants_l2.py`,
  `l2_flip_census_patch.py`, `ic_ranking_trial_patch.py`, `cmp_tbu.py` (each one targets removed lines / names);
  `qa_ic_crossfib_inputs/shadow_ic_changes.py` (its V1 / V123 are the engine since PLAN_P).
- The `/compare` skill: a save-format boundary entry (the IC CSV column `ic_zone_band` after `is_ic_candidate` → keyed
  diff `qa_ic_crossfib_inputs/cmp_trial.py BASE CUR`; the POI meta key; the values `immediate_unfilled_imbalance`
  gone; the MEANING changes of existing columns — `touches_zone` = the touch of the record's band (80–100 on every row
  of a Step 1A record), `is_ic_candidate` reads the immediate gap, `immediate_imbalance_first_c2_idx` IS condition 3
  (its CSV values are unchanged — the CSV always classified every check; landing review #2),
  `has_unfilled_imbalance_after` information only (its values unchanged) — review A #5) + the chart
  counts re-baselined (81/236, 146/135, 289/293).
- Register `IDEA_PARKING_LOT.md` §D session-order item 2 → PLAN_P landed, item 2 done; the D.2 "Step 4" line (more
  records reach it: the all-pinbar ones); memory at `/commit-save`.

---

## 6. The numeric prediction (written BEFORE the landing replay)

**Measured, not derived:** the trial patch applied on HEAD (= PLAN_P's behaviour: the cleanup is output-neutral, §2)
was replayed this session (48.9 s; fetch gate PASS 4228 / 5; skip-grep 22). **The landing replay must reproduce it
EXACTLY:** all 27 CSVs md5-equal to `plan_p_variant/variant_csv_md5.txt`, the three figures equal to
`plan_p_variant/variant_fig_hash.txt` (`plan_m_inputs/fig_hash.py`: H1 `bf695db3…`, counter `9c6baafa…`, confluence
`001a8ccd…`). Any difference = a cleanup bug. Against S_O (`cmp_trial.py`, `cmp_save.py`, `cmp_hover_strings.py` —
saved in `plan_p_variant/`):

- **22 of 27 CSVs differ**; identical: raw, imbalance instances, structure levels, the counter lens's structure events,
  the unresolved triggers (4 → 4).
- **H1** — IC outcomes 4 of 5 records move: sid 0 cycle 0 IC 180 (ranked `exclude_pinbar`) → none
  (`all_ic_candidates_pinbar`); sid 1 cycle 0 IC 694 → none (all pinbar); sid 1 cross 0→1 IC 732 (3A.iv) → none (all
  pinbar); sid 1 cycle 2 IC 832 (3C.ii) → **859** (no pattern, `exclude_pinbar`, zone [0.57786, 0.5785], confirmed 998
  as before) — NOT the 860 the user once expected there (GOTCHAS `:2062-2063`): 859 normal (gap 861), 860 a pinbar
  sharing that gap, 865 pinbar → ranking rule 1 (review B #12; for the H1 chart look). IC 607 unchanged. Drawn POIs **5 → 2**. **One structure event moves: sid 1 cycle 2 CTS_CONFIRMED
  (sd_zone_proximity) 939 → 935** (price 0.57112) — MS's POI inner (a sell POI: the IC's low) 0.57866 (IC 832) →
  0.57786 (IC 859); `final.csv` 14 cells (candles 935–939's stage / threshold / phase + the event); the KL zone of (1, 2)
  confirmed 939 → 935; the H1 WVMI trigger 939 → 935; the fib record (1, 2) `locked_at` 939 → 935. No Step 1A pick on
  H1.
- **Counter lens** — IC outcomes 6 of 7: sub 3 c0 2704 (1A.i) → none (all pinbar); sub 3 c1 2798 (2B.iii) → **2808**
  (3B.iii); sub 3 cross 1→2 2837 (1A.iv) → **2930** (3C.iv); sub 7 c0 4035 (2C.iii) → **4048** (3C.iv); sub 7 cross
  0→1 4051 (2A.iv) → **4131** (3B.iii); sub 7 cross 0→2 4083 (1A.iv, tiebreak) → **4131 (3B.iii, Step 1A)**. POIs 7 →
  6. The cascade from H1's 935: triggers on parent (1, 2) first_counter 3759 → **3743**, subsequent_counter 4159 →
  **4143**; sub 5 [3759, 4159] → **[3743, 4143]**; sub 7 start 4159 → **4143** — before its cycle 1's establishment
  (4157), so its cycle 0 (POI 4048, KL) gets a live stretch: WVMI `cycle_collapsed` True → False. **0 structure-event
  changes.**
- **Confluence lens** — IC outcomes 19 of 20 (the 6 shared with the counter lens's subs 3 / 7 + sub 0 c0 670 → **650**
  (Step 1A, ranked), c1 929 → **923** (3C.iv, Step 1A), c2 1117 (1A.iv) → 1216 (ranked `ic_candle_type`), c3 1660
  unchanged (its tie dissolves: `ip_tiebreak` `immediate_unfilled_imbalance` → none), c4 1772 (2B.iv) → 1778 (only
  candidate); sub 1 c0 1804 → none (`no_ic_candidate`, Step 1A ran), c1 1875 (3C.iv) → 1890 (3D.ii), c2 none → **1934**
  (Step 1A, ranked), c3 2106 (1B.iii) → none (Step 1A ran); sub 2 c0 2383 (3A.iv) → none (all pinbar, Step 1A ran),
  c1 2575 unchanged (`ranking_rule` `immediate_unfilled_imbalance` → `exclude_pinbar`); sub 4 c0 3388 (1A.iii) → 3427
  (3C.iii); sub 6 c0 3809 unchanged (rule → `exclude_pinbar`)). POIs **19 → 16**. Sub 7 start 4159 → 4143 (its
  `probe_finalize_idx` too); the WVMI trigger attribution 939 → 935 (sub 4) and 1039 → 1035 (sub 6, sub 7 ×2), sub 7
  c0 `cycle_collapsed` True → False. **Structure events: 1 meta cell** — sub 1 cycle 1's CTS_CONFIRMED at 1902
  (sd-proximity) `trigger_inner` 0.58074 → 0.58106 (IC 1875's low → IC 1890's); same idx / type / price.
- **The 4 Step 1A picks:** sub 0 c0 650 (never active: its cycle ends at the candle sub 0 goes live — so was the
  baseline's 670), sub 0 c1 923, sub 1 c2 1934, sub 7 cross 0→2 4131 (the last three activate — the band re-check).
- **Charts** — trace / shape counts H1 83/239 → **81/236**, counter 137/123 → **146/135**, confluence 282/276 →
  **289/293**. H1: POIs 180 / 694 / 732 gone, 832 → 859 (`IP 3C.ii …` → `No pattern · ranking rule exclude_pinbar`),
  the fib status `No pattern · IC 180 · …` → `IC: none (all_ic_candidates_pinbar)` and `IP 3C.ii · IC 832` → `No
  pattern · IC 859 · ranking rule exclude_pinbar`, the trigger marker 01-13 01:00 → 01-12 21:00. Lenses: the
  fallback lines (`IC zone 80-100 (fallback)` on 923 / 1934 / 4131; ` · zone 80-100 (fallback)` on those fibs + the
  drawn no-IC fibs where Step 1A ran (`IC: none (no_ic_candidate) · …` — sub 1; `IC: none (all_ic_candidates_pinbar)
  · …` — sub 2 c0)), the two
  tiebreak lines gone (`… immediate unfilled imbalance 1662`, `… most recent before imbalance 4132`), sub 5 / sub 7's
  windows, sub 7's cycle 0 KL / POI / fib now drawn.
- **Unchanged:** fetch gate 4228 / 5, skip-grep 22, unresolved triggers 4, every structure event outside the H1 one.

---

## 7. Tests

The suite on §4's code (measured): 1506 pass, **11 fail + 4 files fail to collect** (they import the removed names);
with the 3 names shimmed (a scratch pytest plugin re-exposing them): **76 fail, 1642 pass** (of 1718) —
`test_plan_l_landing_2` 38, `test_institutional_pattern` 12, `test_plan_k_landing_b` 11, `test_plan_h_landing_b` 9,
`test_plan_l_landing_1` 4, `test_plan_k_landing_a` 1, `test_imbalance_c3_knowability` 1. By cause:

1. **Constructor / meta (mechanical):** `ICCandleCheck` needs `immediate_imbalance_first_c2_idx`
   (`test_plan_k_landing_b::test_an_ic_candidate_passes_every_condition` ×7 — rebuilt on the NEW condition 3: the
   immediate gap present / absent × the reference × the touch, and a row where `has_unfilled_imbalance_after` is True
   but the immediate gap is None → NOT a candidate); `ICSelection` needs `ic_zone_band`
   (`test_plan_l_landing_2::test_the_selection_rejects_inconsistent_fields`); POI metas need `ic_zone_band` (the hover
   tests of `test_plan_h_landing_b` ×6 + `test_m15_lens_sub_poi_hover_is_rendered` + `test_plan_l_landing_2`'s hover
   ×12); the POI meta key list (`test_plan_h_landing_b::test_poi_meta_carries_the_pick_in_place_of_the_variants`).
2. **Fixtures whose IC candidates had only a far gap** (they now fail condition 3): each fixture gets an immediate gap
   (`_far_gap(ic + 1 | ic + 2)`) on exactly the candidates the test needs — keeping each test's PURPOSE (a rank, a
   shape, the reference, the agreed direction, the CTS-anchor rule, the mirror, MS's read …) — or its expectation flips
   where the purpose IS the old condition. Files: `test_institutional_pattern`, `test_plan_k_landing_b`,
   `test_plan_l_landing_1`, `test_plan_l_landing_2`, `test_plan_h_landing_b` (the two resolver tests),
   `test_plan_k_landing_a::test_the_reason_is_reported_and_a_cut_candidate_is_never_picked`,
   `test_imbalance_c3_knowability::test_m1_retired_ms_reads_no_inner_where_fibtracker_creates_no_fib`.
3. **The removed rule's pins — go, or turn into condition-3 / decision-1 pins:** `test_institutional_pattern`'s
   `test_a_sole_immediate_imbalance_ic_is_final_before_the_candle_type`, `test_the_candle_type_runs_over_the_immediate_passers_only`,
   `test_no_immediate_gap_leaves_every_tied_ic_in_play`, `test_a_filled_gap_is_neither_immediate_nor_the_reference`
   (→ a filled immediate gap fails condition 3), `test_the_immediate_gap_may_sit_at_the_cts_anchor` (→ condition 3 at
   the anchor), `test_the_users_sequence_…` (→ the new sequence's pick); `test_plan_l_landing_1::
   test_the_immediate_imbalance_filter_on_two_input_lists` (the helper is gone); `test_plan_l_landing_2`'s ranking
   cases `one_immediate` / `one_immediate_of_three` / `runner_up_not_a_pinbar` / `no_immediate`,
   `test_pinbars_are_excluded_before_the_immediate_imbalance`, `test_the_shared_helpers_receive_each_rule_sets_own_list`
   (monkeypatches the removed helper — keeps the candle-type helper), hover case `meta2`;
   `test_plan_k_landing_b::test_the_tiebreakers_run_over_the_surviving_tied_ics_only`; `test_plan_l_landing_2::
   test_the_ranking_constants_are_the_decided_words` (`:412`, imports the removed constant — collection).

**New — `tests/test_plan_p_ic_condition3_step1a.py`** (fixtures where the OLD and NEW rules DIFFER — last session's
lesson):
- condition 3: a candle with only a far gap (old condition passes) is NOT a candidate, `has_unfilled_imbalance_after`
  True on its row; the same candle with a gap at IC + 2 is; IC + 3 is not (the window); a filled immediate gap fails;
- the pick never reads `has_unfilled_imbalance_after` (an immediate gap implies it, so the far-gap-only case is THE
  distinguishing fixture; the CSV row carries both columns);
- Step 1A: no 61.8–80 candidate + an 80–100 candle with an immediate gap → picked, band 80-100, on the POI meta, every
  CSV row, the POI hover line and the fib status suffix; a 61.8–80 all-pinbar record does NOT run Step 1A (an 80–100
  maru with an immediate gap present → still `all_ic_candidates_pinbar`, band 61.8-80); Step 1A keeps conditions 2
  and 3 (an 80–100 candle failing the reference / without an immediate gap → `no_ic_candidate`, band 80-100); a
  record with no checks → band 61.8-80; the fallback zone is 80–100 and not 61.8–100 (a candle touching only 61.8–80
  but failing condition 3, and one touching only 80–100 passing → the 80–100 one);
- the activation: a Step 1A POI activates on the AS-OF 80–100 touch (it would never touch 61.8–80); a 61.8–80 POI is
  unchanged; the activation's condition 3 = ANY gap (the IC's immediate gap commit-filled, a later gap unfilled → the
  POI stays active — decision 4);
- decision 1: the tiebreak rule 2 picks the oldest tied IC (reference = its immediate gap); ranking rule 3 picks the
  oldest in play, or its neighbour sharing that gap (the closer one);
- MS's read returns a Step 1A IC's inner (`poi_inner_for_fib`);
- `ICSelection` rejects an unknown band; `ic_zone_config` maps the bands (and the primary returns `config` itself).

**As built (after the plan review):** the 76 reworked as above (+ review B #5: the two `apply_tiebreak_rules` callers
of `test_plan_l_landing_1` lose the positional dict; `test_hover_tiebreak_lines`' immediate case → a REJECTION;
`test_plan_l_landing_2`'s CSV column-order pin gains `ic_zone_band`; the passing-but-stale FIX comments / rule
numbers rewritten); the shared M1 frame (`test_plan_h_landing_b._m1_frame`) gains the immediate gaps 33 / 37 (far
below every later low); `test_imbalance_c3_knowability`'s M1 non-vacuity check reads a COPY with 32's immediate gap
(on the frame itself that gap would form before the establishment and create the fib); `test_fib_chart_render._ip`
gains `band` (default `61.8-80` — the hovers assert it). New file `test_plan_p_ic_condition3_step1a.py` (+ review B
#6: the fallback line on a pattern AND a ranked IC, the fib suffix on a pattern / ranked pick, a no-IC row and absent
on `not evaluated`, both hovers reject an unknown band, Step 1A with a pattern pick; review B #1: the activation case
where 80–100 and 61.8–100 differ). Already pinned (B #6 c / d): the window clipped at the CTS anchor
(`test_immediate_window_stops_at_the_cts_anchor`), the fill horizon (`test_the_fill_horizon_is_the_cts_anchor`).

**Mutants** (`mutant_runner.py`, the full suite each; `qa_ic_crossfib_inputs/mutants_plan_p.py`; on the landed code):
`is_ic_candidate` reads `has_unfilled_imbalance_after` (old-for-new); condition 3 dropped; Step 1A never runs; Step
1A also after an all-pinbar Step 3; Step 1A on the primary band; Step 1A drops condition 2 / 3 (touch only); the band
not recorded (always primary); the activation's touch on `config` (primary) instead of `ic_zone_config`; the
fallback bounds 61.8–100 / 80–80; the POI hover line missing / on the primary band / dropped from the pattern
branch; the fib suffix missing; the band asserts deleted; the CSV column not exported; the most-recent rule's
reference after the NEWEST candidate; the activation's condition 3 counting only the IC's immediate gap; the
condition-3 window's CTS-anchor clip dropped. **Expected equivalent** (survive by construction, not gaps — review B
#7): `is_ic_candidate` additionally ANDing `has_unfilled_imbalance_after` (an immediate gap implies it: same sd
filter, same fill horizon, a sub-window); re-adding the immediate filter to either rule set (a no-op); the 61.8–100
band ON THE PICK alone (§2).

---

## 8. The landing

1. §4's code (built) + §7's tests + §5's docs; the full suite from the repo root.
2. Replay (`python -m engine_v2.run_replay > run.log 2>&1`): fetch gate, skip-grep, timing.
3. `/compare` vs S_O: md5 / figure-hash equality to the variant (§6) — then the keyed `cmp_trial.py` +
   `cmp_hover_strings.py` vs S_O, every delta checked against §6.
4. Landing review (read-only, size + tokens stated first); mutants.
5. The user's chart look (H1 first) → commit on the user's go → `/commit-save` (S_P = the new `LATEST_<branch>`).

## 9. Review log

**Plan review, 2026-10-13** — 2 lenses = **600,960 tokens**: lens A (mechanism / readers / prediction, read-only on
the main tree: 292,920 tokens, 75 tool calls — verified all 27 CSVs md5-equal to the variant, the 3 figure hashes,
≥ 12 §6 numbers, the byte-identity claim semantically against the trial patch); lens B (code / tests / docs /
mutants, a worktree reset to `d8f6279` + the code patch: 308,040 tokens, 101 tool calls — the suite as patched
(13 failed = the 11 + 2 environment; with the shim 78 = 76 + the same 2, per-file counts = §7), 7 mutants on the
pre-rework suite). No BLOCKER.

| # | Sev | Finding | Disposition |
|---|---|---|---|
| B1 / A1 | MAJOR / MINOR | §7's "80–100 not 61.8–100" pin cannot tell them apart on the pick (M9 SURVIVED the pre-rework suite) | §2 states the equivalence; pinned where they differ: `test_plan_k_landing_b`'s Step 1A case (32 / 36 touch 61.8–80, not 80–100 → `touches_zone` False) + the new activation case |
| A2 | MINOR | A Step 1A IC inside the as-of 61.8–80 zone is inactive (decision 2's consequence; 0 on the window) | §2 + POI_ZONES_SPEC §4 "Lifecycle"; told to the user |
| A3 | MINOR | The BOS anchor candle always touches 80–100 (IC 1934 = its BOS anchor) | asked: the user "Keep it" (decision 7); spec §3.1 |
| A4 | MINOR | MS's Step 1A reads unmeasured | `shadow_plan_p_ms_reads.py`: H1 0 / 56, M15 43 / 429 (§2) |
| A5 | MINOR | The /compare boundary misses the meaning changes of existing columns | §5 |
| A6 / B2 / B8 | MINOR / NIT | §5 missed doc sites / superseded pointers / stale tools | §5 (the plan-review additions line) |
| A7 | NIT | §2's "attrs are start-sorted" reason | reworded (§2) |
| A8 | NIT | `qa_findings.md` cites §3 for the prediction; its `:121` "confluence triggers unchanged" (sub 7's `probe_finalize_idx` moves) | fixed |
| A9 | NIT | The fib box stays 61.8–80 on a Step 1A record | §2 + CHARTING_SPEC |
| A10 / B9 | NIT | Assert the fallback band ⇒ checks | added (`ICSelection`); decision 1 as a runtime check — not taken (pinned by tests) |
| B3 | MINOR | Stale comments outside the patch | §4 item 4 |
| B4 | MINOR | "Fails loudly" overclaimed for `ICCandidateRecord.ic_zone_band` (default None) | both hovers assert the band (§4 item 2) |
| B5 | MINOR | §7 categorisation gaps + passing-but-stale tests | §7 "As built" |
| B6 | MINOR | Missing pins (pattern-branch line, suffix variants, rejections, Step 1A + pattern, written CSV) | added; c / d already pinned |
| B7 | NIT | Expected-equivalent mutants unnamed | §7 |
| B10 | NIT | Docstring gaps (CTS clip, `derive_ic_candidates`, the POIZone meta comment) | fixed |
| B11 | NIT | No timing comparison | at the landing (§10) |
| B12 | MINOR | H1 sid 1 cycle 2 picks 859, not the GOTCHAS-expected 860 | §6 + GOTCHAS; for the H1 chart look |

Confirmed by the lenses (unchanged): §2's mechanism (a)–(f) at the code; the removed rule a no-op (counts,
runner-ups, the 4th tuple element unchanged); Step 1A's scope; the production callers / constructors; the
byte-identity of the landing replay to the variant (the cleanup removes unreachable branches, swaps `.get` /
`getattr` for direct reads of always-present values, adds asserts, moves a keyword-built field — docstrings
otherwise).

## 10. The landing, as measured (2026-10-13, uncommitted on `d8f6279`)

- **Suite:** 1733 passed (1718 − 3 removed immediate-rule ids + 1 distinguishing row in `test_plan_k_landing_b` + 17
  in the new `test_plan_p_ic_condition3_step1a.py`), 143 s.
- **Replay:** fetch gate PASS (4228 / 5), silent-skip grep 22 (= S_O), timing 91.0 s wall (multi_tf_dual 17.8, charts
  11.5 / 14.3 / 40.8) — every stage ~2× the variant's 48.9 s, data fetch and candle features included: machine
  variance; S_O's own landing replay took 73.5 s (PLAN_O §10). The condition-3 scan on every check (MS's read too)
  shows no stage-specific cost.
- **`/compare` vs S_O = §6 EXACTLY:** all 27 CSVs md5-equal to the variant (`plan_p_variant/variant_csv_md5.txt`), the
  3 figures hash-equal (`variant_fig_hash.txt`), the keyed diff (`cmp_trial.py`) identical line for line to
  `cmp_trial_vs_S_O.txt`; chart counts 81/236, 146/135, 289/293.
- **Mutants** (`qa_ic_crossfib_inputs/mutants_plan_p.py`, 28, the full suite each, in parallel): **27 / 27 KILLED** — condition 3 (P01–P03), Step 1A (P04–P13), the consumers (P14–P20), the hovers (P21–P27); thinnest kills: P03 (the CTS-anchor clip, 1 test), P13, P14, P21, P23–P25, P27 (1 test each). **E01 SURVIVED as predicted** — equivalent (an immediate gap implies the retired any-gap test).
- **Landing review (2026-10-13)** — 1 read-only lens, **384,641 tokens**, 96 tool calls (est. ~250k): no BLOCKER / MAJOR; re-verified the md5 / figure-hash identity, §6 record by record, every doc number, a failing test for every listed mutation. Folded: #1 the `/compare` skill entry + count block and the register's item-2 line — in the commit (the skill entry at `/commit-save`, with the save name, as PLAN_O); #2 `immediate_imbalance_first_c2_idx` was NOT "classified checks only" in the CSV (`classify_all`): 0 value differences in that column and in `has_unfilled_imbalance_after` vs S_O — only their roles changed (spec §3.5 + §5 corrected); #3 the reference-cut tests could no longer show a MOVE on `_tb2_frame` (a tie now goes to the oldest) — re-pinned on the best-rank frame (9 → 7); #4 IMBALANCE_FILL_SEMANTICS `:195-198` + `patterns/imbalance.py` docstring; #5 the CSV test asserts the condition-3 column; #6 / #7 wording ("or Step 1A", the §2 cite, this status, `qa_findings.md`'s list order).
- **The user's chart look (2026-10-13):** "Charts look good, go ahead and commit once the review is done".
