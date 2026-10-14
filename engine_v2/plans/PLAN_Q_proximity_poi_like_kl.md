# PLAN_Q — the zone-proximity scan reads a POI exactly like a KL zone (the user, 2026-10-14)

**Status: LANDED 2026-10-14** (plan review + landing review folded in, §9; the user's chart look: "Charts look good";
the commit + save: §10) — baseline **S_P** `artifacts/commits/week9-entries-decisions/20261013_231327_23f395d`
(= `LATEST_week9-entries-decisions`; source `23f395d`); HEAD at drafting `14d5491`; suite 1733. Session order item 3
(register `IDEA_PARKING_LOT.md` §D "SESSION ORDER": the opposite-direction zone-proximity trigger — changes and edits).
This plan is the item's first landing, on the sd side, ahead of the user's opposite-direction changes (the user,
2026-10-14: "Yes, write PLAN_Q and build it now"). One cause: the POI read of `check_zone_proximity`.

---

## 1. The rule

**The user (2026-10-14), verbatim:**
- "it should not be the case that if the POI had existed (even if inactive) it is not used for sd proximity trigger."
- On the choice between dropping only the lifecycle clamp (A) or ignoring the POI's activation entirely (B): "For the
  purpose of proximity triggers, I would like POI to be treated exactly like a KL zone, so in this case I think that
  would be B".

**Engineering reading.** The scan reads the BOS and CTS KL zones by LOCATION: `_find_kl_zone_for_cycle` matches
(structure, cycle, kind) and reads only `meta["inner"]` — never `confirmed_idx`, `end_idx`, `status` or
`activation_history`; the zone counts at every candle of the cycle's scan window `[CTS_CONFIRMED moment,
min(next BOS_CONFIRMED moment − 1, the structure's realised reversal − 1, last candle)]`. From this plan a POI is read
the same way:

- **A POI of the cycle counts as an sd zone at EVERY candle of the scan** — from the scan start, like the BOS zone.
- **Its activation history is not read** — neither the lifecycle clamp (a collapsed cycle's POI: empty history) nor the
  off-stretches its own imbalance condition produces (`imbalance_filled` → `conditions_re-met`).
- **Both of its bounds provably lie outside the scan (§3) and are ASSERTED, not read:** its unclamped activation
  floor `poi_unclamped_activation_floor_idx(poi)` = `max(meta["cts_established_idx"], ic_idx[,
  meta["pattern_confirmed_idx"]])` (POI_ZONES_SPEC §4 "Activation floor" without the structure lifecycle-start term —
  one helper) ≤ the scan start; its `end_idx` > the scan's last candle. (Drafted as a silent gate `i >=` the floor;
  the plan review's finding 2 — the gate never binds on the main, and "exactly like a KL zone" counts from the scan
  start — made it an assert: a future POI that breaks the invariant fails loudly instead of being read by a rule
  nobody decided.)
- **Consequence (the plan review's finding 10):** on a collapsed cycle a POI can now take the cycle's first sd trigger
  — that trigger's `zone_kind` / `trigger_inner`, the main WVMI gate record's attribution (`triggered_by_event_idx`
  stays the candle) and the later triggers' `prior_sd_zone_kind` follow it (S_O's (1,1): 820 sd:BOS → sd:POI).
- **Unchanged:** the POI's own activation history and everything that reads it (the chart fill tints, `status`, the POI
  CSV, the POI hover); MS's sd-proximity CTS confirmation (it already reads the cycle's IC pick with no activation gate
  and no lifecycle clamp — `market_structure.py` `_refresh_poi_inners_for_cycle`); the KL read; the alternation, the
  narrow-cycle rules, the scan window.

---

## 2. Why — the measured asymmetry (S_P / S_O, the 2026-10-14 discussion)

- **Collapsed H1 cycles still fire triggers.** Structure 1 becomes known at 902 (structure 0's realised reversal,
  `STATE_CHANGED(to=reversal)` at 902) = its lifecycle start; its cycles (1,0) (established 703) and (1,1) (established
  748) both have the empty lifecycle `[902, 902)`. Their KL zones are `inactive`, `confirmed_idx` 902, `end_idx` 902,
  activation history `[]` (S_P `_kl_zones.csv`) — yet the scan fires on (1,1): **820 sd:BOS** (high 0.58064 vs the BOS
  inner 0.58132, 6.8 pips short) and **871 opp_sd:CTS** (low 0.57604 vs the CTS inner 0.57538, 6.6 pips short). The
  hover shows `zone_confirmed_idx=902` on both — the zone's clamped lifecycle start, after the trigger candle. The M15
  subs these triggers would start are NOT built (LANDMINES "Collapsed Parent Cycles: Log the Trigger, Don't Build the
  Sub"; S_P `_M15_unresolved_triggers.csv`: first_counter 3283 = LOH(820), subsequent_confluence 3487 = LOH(871)).
- **A POI on a collapsed cycle never could.** The POI gate was `poi_active_as_of(pz, idx)` (`zone_proximity.py:496`),
  which walks the POI's activation history; the history's first activation is floored at the structure's lifecycle
  start (`poi_zones._compute_poi_activation_history`, `lifecycle_floor_idx`), so on a collapsed cycle it is empty.
  S_O's (1,1) POI — IC 732 (2025-12-30 10:00, a pinbar), cross fib 689 → 761, institutional pattern 3A.iv, zone
  [0.58068, 0.58118], `inactive`, history `[]` — was ignored while the equally inactive BOS zone of the same cycle was
  used. Counted like the BOS zone, 820 would have read **sd:POI** (inner 0.58068, the nearer one) — same candle: the
  trigger needs a high ≥ 0.57988 and the highest of 810–819 is 0.57968 (819). S_O's (1,0) POI (IC 694) changed
  nothing: its trigger needs ≥ 0.58212, the (1,0) scan window 727–747 peaks at 0.58196 (728). PLAN_P removed both POIs
  (IC condition 3: 732's immediate gap, first gap candle 733, fills at 740, before the CTS anchor 761).
- **Origin — a side effect, never a decision.** The POI gate on the scalar `confirmed_idx` (end-bounded) dates from
  `6c7f1a3` (2026-02-16, `wvmi.py check_proximity_activation`); `677997b` (2026-04-28) extracted it into this module
  (the plan review's finding 7); `e352ef8` (2026-05-26 morning, the lifecycle-start clamp — "so no zone activates before its
  structure is alive") floored every zone's first activation at the structure's lifecycle start: KL's `confirmed_idx`
  too, but the scan never read it, so only the POI read went dark on collapsed cycles; `09ee05e` (2026-05-26
  afternoon) moved the POI gate to the per-candle history (to see the earlier active stretches of a flapping POI) —
  the exclusion carried over. (A gitignored `/prepare` report, `artifacts/_prep_reports/prov_provenance.md`, 2026-08-11,
  noted the asymmetry; no canonical doc did.)
- **(0,1) vs (1,1):** the one differing term is the lifecycle start — 115 for structure 0 (its first CTS
  established), far below (0,1)'s other terms (CTS established 652, IC 607, pattern confirmed 610 → active from 652, the
  748 trigger is sd:POI); 902 for structure 1, which is also (1,1)'s end.

---

## 3. Invariants (traced; the code asserts both)

- **A POI's unclamped activation floor is at or before its cycle's scan start (the CTS_CONFIRMED moment).** Measured:
  (0,1) 652 vs 683; (1,2) 902 vs 935 (S_P); S_O's (1,1) 748 vs 810, (0,0) 180 vs 439. The chain (the plan review's
  finding 1, which corrected the draft's): the CTS-established moment precedes the confirmation; the IC is at or before
  the fib's FINAL `cts_idx` (the leg scans are cut at it — `patterns/institutional_pattern.py:151-160`, `select_ic` passes
  `fib_state.cts_idx`); `pattern_confirmed_idx` ≤ `cts_idx` + 1 (`patterns/institutional_pattern.py:164`); and the CTS anchor
  is strictly before the CTS_CONFIRMED moment (a proximity confirmation needs `i > st.cts.idx`,
  `market_structure.py:1914`; a pullback follows the anchor; S_P: 0 of 30 CTS_CONFIRMED with anchor == moment, 0
  CTS_UPDATED after a CTS_CONFIRMED — the CTS extends only until it is confirmed). The IC CAN lie past the
  ESTABLISHMENT (the draft's chain said otherwise — wrong): S_O's (0,0) IC 180 vs established 115, S_O's (1,1)
  established 748 vs anchor 761 with IC candidates to 758 — so the IC term binds in the floor (and in the activation
  sweep), never past the scan start. Asserted per POI at the scan.
- **A POI's `end_idx` is always past the scan's last candle.** The POI end = its cycle's lifecycle end
  (`compute_cycle_lifecycle`): `min(next cycle's clamped start, the realised reversal)` (no cap on the main); the next
  cycle's clamped start ≥ its `CTS_ESTABLISHED` moment == its `BOS_CONFIRMED` moment (the definitional identity,
  `parent_tables.py:157`) > the scan's `next BOS_CONFIRMED moment − 1`; the realised reversal (the same
  `compute_reversal_idx_by_sid`) > the scan's `reversal − 1`. So ignoring it changes nothing — asserted per POI at
  the scan (a future POI end inside the scan would otherwise be silently ignored).
- **The zone proximity scan runs on the MAIN only** (`orchestrator.py` step 9: `wvmi == "first_sd_prox"`) — no sub
  path changes.

---

## 4. The code (as built — the plan review's finding 2 folded in)

1. `zones/poi_zones.py`: extract `poi_activation_floor_idx(cts_established_idx, ic_idx, *, pattern_confirmed_idx,
   lifecycle_floor_idx) -> int` — the floor's four terms (keyword-required, no default, like the activation sweep's
   parameter, PLAN_L A7); `_compute_poi_activation_history` calls it (output-neutral). POI_ZONES_SPEC §4 names it.
2. `zones/zone_proximity.py`:
   - `poi_unclamped_activation_floor_idx(pz) -> int` = `poi_activation_floor_idx(pz.meta["cts_established_idx"],
     pz.ic_idx, pattern_confirmed_idx=pz.meta["pattern_confirmed_idx"], lifecycle_floor_idx=None)` (direct indexing —
     both keys are stamped by the only `POIZone` constructor, `poi_zones.py`). Drafted as `poi_proximity_start_idx`;
     renamed once it stopped being a start (the review's finding 2).
   - `check_zone_proximity`: per cycle, after the scan window, every POI of the cycle asserted: the unclamped floor
     `<= scan_start`, `end_idx` `None or > scan_end` (§3).
   - `_try_trigger_at_candle`: the `if not poi_active_as_of(pz, idx): continue` gate is deleted — every POI of the
     cycle joins the sd inners at every candle; the `poi_active_as_of` import goes.
   - Docstrings: the module's ("Zones are read by LOCATION, never by lifecycle": KL + POI, the asserts, the
     collapsed-cycle consequence, the subs not built), `check_zone_proximity`'s `poi_zones` parameter.
3. The two trigger hovers (`charting/export_plotly.py` `_poi_confirmed_idx`, `charting/export_m15_chart.py` its copy):
   a POI marker's `zone_confirmed_idx` = `poi_unclamped_activation_floor_idx(best)`; was `poi_confirmed_idx_as_of(best,
   at_idx)` (the active stretch's start, −1 outside one — meaningless once a POI can fire in an off-stretch or on a
   collapsed cycle); the `at_idx` parameter and the `poi_confirmed_idx_as_of` imports go. The KL markers keep the KL
   zone's lifecycle-clamped `confirmed_idx` — the two meanings are documented (CHARTING_SPEC §12) and the KL side's
   option is a register §G line (the review's finding 6; changing it = a second cause).
4. `zones/poi_lifecycle.py`: `poi_active_as_of` / `poi_confirmed_idx_as_of` lose their last engine caller — KEPT
   (the documented per-candle query, LANDMINES "POI Activation Is Per-Candle"; their tests stay); the module
   docstring's "imported by" list updated.

No event, meta key, CSV column or chart element is added or removed → no save-format boundary.

---

## 5. Docs (same commit)

- The module docstrings (`zone_proximity`, `poi_lifecycle`, the floor helper's).
- LANDMINES: "Zone proximity gate" (item 4, `:360`: "Uses only active POI zones at each candle" → the KL-like read +
  the asserts + the collapsed-cycle consequence); "POI Activation Is Per-Candle, Not a Scalar Span" (the proximity gate
  is no longer a consumer; `:1832`'s "chart fills, the proximity gate, and hover labels" — review finding 4);
  "Collapsed Parent Cycles" (a paragraph: the parent itself is built in full — its zones, WVMI record and proximity
  triggers exist — and every no-build case: a collapsed parent cycle's sub, a same-key re-trigger, `pending` /
  `probe_failed` / `geometry_failed`).
- GOTCHAS: "POI `confirmed_idx` Is a Lossy Scalar" (a dated line); `:668` "only use active POI zones at each candle
  (check `confirmed_idx <= candle <= end_idx`)" (review finding 4).
- GLOSSARY: "collapsed cycle" (its proximity triggers fire and are drawn; the subs they would start are not built);
  "sd zone" (a POI counts like the BOS zone).
- POI_ZONES_SPEC §4 "Activation floor" (the helper, its two callers, the scan's assert) + the `confirmed_idx` field
  notes (`:720`, `:834`).
- FIB_LIFECYCLE_SPEC `:380` ("walked per-candle by the proximity gate" — review finding 4); ARCHITECTURE `:250`.
- CHARTING_SPEC §12 (NEW): the zone-proximity trigger markers — placement, the hover fields, `zone_confirmed_idx`'s
  two meanings (review finding 6). Register §G "Overall chart revamp" item (2): the KL marker's clamped value.
- Not touched: MARKET_STRUCTURE_SPEC `:312` "active POI inners" (the review: optional, already stale — MS reads the
  representative fib's IC pick, no activation; not this plan's cause).
- At `/commit-save`: the `/compare` skill's PLAN_Q entry; the register §D item 3 line; memory.

---

## 6. Prediction (`/compare` vs S_P), written before the variant replay

**Engine:** identical. The six H1 triggers are unchanged — (0,1) 748 sd:POI; (1,1) 820 sd:BOS, 871 opp_sd:CTS; (1,2)
935 sd:POI, 954 opp_sd:CTS, 1035 sd:POI — because no POI sits on a collapsed cycle in S_P and the only POI
off-stretches, (1,2)'s 952–953 and 992–997 (history `[902 A, 952 D, 954 A, 992 D, 998 A]`), cannot fire: during
952–953 the scan expects an opp_sd; during 992–997 it expects an sd, which needs a high ≥ 0.57786 − 8 pips = 0.57706,
and the highest high of 955–1034 is 0.57674 (1021).

- **All 27 CSVs md5-equal to S_P.**
- **Figures: exactly 3 changed values, one per chart** — the 1035 marker's `zone_confirmed_idx` (customdata
  element 8) **998 → 902** (the (1,2) POI's unclamped activation floor: max(CTS established 902, IC 859;
  ranked — no pattern term)):
  H1 (`idx` 1035), M15 confluence and M15 counter (`idx_15M` 4141). The other five markers keep theirs (748: 652 →
  652; 935: 902 → 902; the KL markers 902 / 902 / 935 unread by this change). Every other trace, shape and the
  layout identical; chart counts 81/236, 146/135, 289/293.
- Fetch gate 4228 M15 candles / 5 chunks; silent-skip grep 22.

**Measured by a variant replay** (the §4 engine + chart code on HEAD, no docs / tests): its 27 md5 + 3 `fig_hash.py`
hashes saved in `plan_q_inputs/` BEFORE the landing; the landing replay must equal them exactly. (The variant ran the DRAFTED
silent gate; the as-built assert — §4 — must reproduce it too: the gate never bound.)

---

## 7. Tests

`tests/test_zone_proximity.py` (the fixture `_poi_zone` takes `cts_established_idx` keyword-required + `ic_idx` /
`pattern_confirmed_idx`, stamps both meta keys and the scalar `confirmed_idx` = the history's last activate):
- Rewritten: `test_poi_not_active_yet_excluded` → `test_poi_floor_at_the_scan_start_counts_from_the_scan_start` (the
  `<=` boundary) + `test_poi_floor_after_the_scan_start_raises`. The old test pinned NOTHING (the review's finding 8):
  its candle sat 10 pips from the POI inner, outside H1's 8 since the 2026-05-28 tuning, so it passed whatever the
  gate did. `test_poi_suppressed_during_inactive_stretch` → `test_poi_counts_during_inactive_stretch`.
- Renamed: `test_poi_included_in_sd_inners_when_active` → `…_nearer_than_bos`. Kept:
  `test_poi_fires_in_first_active_stretch_not_last_activate` (now: neither the scalar 20 nor the history is read).
- New: `test_poi_on_collapsed_cycle_counts_like_its_bos_zone` (S_O's (1,1) numbers: sd −1, BOS inner 0.58132, POI
  bottom 0.58068, history `[]`, the candle's high 0.58064 → sd:POI at 0.58068);
  `test_poi_unclamped_activation_floor_idx_is_the_floor_without_the_lifecycle_term` (ranked / pattern / late pattern /
  an IC past the establishment — S_O's (0,0): 115 vs IC 180); `test_poi_end_inside_the_scan_raises` parametrized (end
  10 and 19 = the last candle raise; 20 = the next candle and None pass — the boundary, review finding 5).
- `tests/test_poi_activation_moment.py`: `test_poi_activation_floor_idx_takes_the_max_of_its_terms` (the four terms,
  `None`s skipped, keyword-required; the IC term binding — 748 vs IC 753 → 753).
- `tests/test_proximity_trigger_hover.py` (NEW — review finding 3: on S_P every POI's floor == its establishment, so the
  landing replay cannot catch a hover that shows the establishment): the H1 chart (`export_chart_plotly` on the
  `test_plan_h_landing_b` pipeline fixture, a POI whose floor is a late pattern) and the M15 overlay
  (`_render_proximity_triggers_overlay`, a POI whose IC 3 is past its establishment 2) — the marker's customdata[8] ==
  the floor.

## 8. Mutants (`plan_q_inputs/mutants_plan_q.py`, `mutant_runner.py`, full suite)

Must be KILLED: Q01 the gate restored to `poi_active_as_of`; Q02 the floor assert removed; Q07 its boundary `<=` →
`<`; Q05 the floor read from the activation history (the clamp leaking back — a collapsed POI: never); Q10 the floor
from the lossy scalar `confirmed_idx`; Q06 the end assert removed; Q11 its boundary `>` → `>=`; Q03 / Q04 / Q08 the
pattern / lifecycle / IC term dropped from `poi_activation_floor_idx`; Q12 / Q13 the activation sweep's call dropping
the lifecycle / pattern term; H01 / H02 a hover (H1 / M15) returning `meta["cts_established_idx"]`. (No Q09:
the drafted "`lifecycle_floor_idx` passed from the start helper" mutant cannot be built — plan review finding 9.)
Q08 SURVIVED the first run: the draft called it equivalent ("the IC precedes the CTS anchor ≤ the establishment") —
wrong (§3: the IC can lie past the establishment); no test had pinned the term, before PLAN_Q either.

## 9. Reviews

**Plan review — 1 lens, 227,056 tokens** (68 tool uses, ~12 min; read-only, no worktree; it ran HEAD's
`check_zone_proximity` on the S_P and S_O CSVs under the old gate, this rule and an always-counting POI — the six
triggers reproduce under all three, the 3 hover cells 998 → 902 reproduced). Verdict: sound, the prediction checks
out; 4 SHOULD-FIX + 6 NIT, all folded in:
1. §3's chain wrong (the IC can lie past the establishment) and Q08 not equivalent → §3 rewritten, Q08 pinned.
2. "the moment the POI exists" overstated, the start gate never binds on the main → the gate became an assert (§1,
   §4), the helper renamed `poi_unclamped_activation_floor_idx`.
3. The hover pins could not catch a wrong floor → `test_proximity_trigger_hover.py` + mutants H01 / H02.
4. Three more statements made false → GOTCHAS `:668`, FIB_LIFECYCLE_SPEC `:380`, LANDMINES `:1832` (§5).
5. The end-assert boundary unpinned → parametrized test + Q11.
6. The hover's two meanings → CHARTING_SPEC §12 + register §G (2).
7. The gate's origin is `6c7f1a3` (2026-02-16), `677997b` only extracted it → §2.
8. `test_poi_not_active_yet_excluded` never tested anything → §7.
9. The "lifecycle_floor_idx passed from the start helper" mutant cannot be built (a POI carries no structure start) →
   Q05 reads the history instead.
10. The consequence on a collapsed cycle (the first sd trigger, the WVMI attribution) → §1.

**Landing review — 1 lens, 178,692 tokens** (56 tool uses, ~9 min; read-only). Re-checked the 27 md5, the 3 leaves,
the 14 mutants, CRLF on every file; verdict: correct, output-neutral, well pinned. 3 SHOULD-FIX (wording) + 6 NIT,
all folded in: the `check_zone_proximity` docstring's "BOS or active POI" + `_zone_render.py:135`; GOTCHAS `:1527-1537`
(the `poi_confirmed_idx_as_of` bullet still said the hover shows it); "the pattern confirms by it" → "by the anchor +
1" (POI_ZONES_SPEC, the helper's docstring, a test docstring); LANDMINES "Collapsed Parent Cycles" (a collapsed
cycle's main WVMI record exists only when its first sd trigger fires); the POI_ZONES_SPEC paragraph moved after its
"where" clause; the boundary test strengthened (candle AT the scan start, an empty history — the old gate fails it)
and renamed `test_poi_counts_from_the_scan_start_candle_with_its_floor_at_it`; the hover test imports
`_make_ip_multicycle_data` from its home; the hovers' nested helper renamed `_poi_activation_floor_idx`; this status
line, the `patterns/` paths, the Q09 note. Left out of the landing (not this cause): `market_structure.py:1875` /
MARKET_STRUCTURE_SPEC `:312` "active POI" — "active" meant the representative fib, MS reads no POI activation;
reworded after the save in its own docs-only commit (the user, 2026-10-14: "let's go with your recommendation").

## 10. Landing record

**Built 2026-10-14 (uncommitted; final code = both reviews folded in):** suite **1743** passed (1733 − 2 rewritten +
12 new, incl. the parametrized end test's 4 cases); the landing replay on the final code == the variant snapshot (27
CSV md5, 3 `fig_hash.py` hashes — `plan_q_inputs/variant_*`) = §6 exactly: vs S_P 0 / 27 CSVs differ, 3 figure
leaves (`zone_proximity:trigger` customdata[5][8] 998 → 902 on each chart), layouts equal, counts 81/236, 146/135,
289/293; fetch gate 4228 / 5 chunks; silent-skip grep 22; replay 45–70 s wall (70 s with a mutant run in parallel).
Mutants: run 1 (the drafted gate) 10 / 11 (Q08 survived → pinned); run 2 Q08 KILLED; run 4 (the final code)
**14 / 14 KILLED** (`plan_q_inputs/mutants_run*_results.json`). Reviews: plan 227,056 tokens, landing 178,692.
The user's chart look (H1 first, then both M15 lenses), 2026-10-14: "Charts look good" → committed and saved with
`/commit-save`: source `da8c15e`, save **S_Q** `20261014_170919_da8c15e` (save commit `ff8d246`, trunk `ab8e3cf`;
= `LATEST_week9-entries-decisions`); the save replay md5- / figure-hash-equal to the variant again (27 / 27, 3 / 3;
48.5 s wall after one OANDA 401 on the first try); suite 1743 on the commit.
