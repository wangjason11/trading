# PLAN_O — the IC reference's CTS = cycle T−1's for every fib record (the user's change 2, 2026-10-13)

**Status: LANDED 2026-10-13** — source `ce5ac9b`, save **S_O** `20261013_151425_ce5ac9b` (save commit `b138424`, trunk
`07f5ff6`; = `LATEST_week9-entries-decisions`, PLAN_P's baseline). Plan review 2 lenses (387,076 tokens, §9); suite
1718; `/compare` vs S_N_hover = §6 except one H1 hover line (§10); mutants 16 / 16 KILLED + 1 equivalent; landing
review 1 lens (218,728 tokens, §10); the user's chart look: "Charts look good". Baseline **S_N_hover**
`artifacts/commits/week9-entries-decisions/20261012_194022_9f9406d` (= `LATEST_week9-entries-decisions`); HEAD at
drafting `caf0f01`; suite 1712. Session order item 2 (register `IDEA_PARKING_LOT.md` §D "SESSION ORDER"): the IC
identification + cross-cycle fib questions and edits.

Inputs (`qa_ic_crossfib_inputs/`): the user's write-up verbatim `user_writeup_20261013.md` (three changes); the Q&A +
the measurements `qa_findings.md`; the read-only shadows `shadow_ic_changes.py` (all three changes, first order) and
`shadow_change2.py` (this change alone: every caller of the reference + MS's proximity checks); the combined TRIAL's
change-2 part `trial_p1_change2.patch` (the starting point of §4's code, not a landing).

The user's landing split (2026-10-13, verbatim in `qa_findings.md`): **change 2 alone first** (this plan: its own
`/compare` + commit-save — "Change 2 I think we will for sure keep"), then changes 1 + 3 in ONE commit (PLAN_P, to
write — the revert unit).

---

## 1. The rule

**The user (write-up, 2026-10-13):** "In reality, regardless if we have cross fib 0-3, 1-3, or 2-3, the true CTS
reference in all these cases should actually be cycle_2 CTS. From a breaking previous CTS perspective, we do not treat
the cross fib cycles as unified. Similarly, if we have cross fib 0-1, cycle_0 CTS would be the CTS reference; if we
have cross fib 1-2, cycle_1 CTS would be the CTS reference. The CTS reference will always be the max cycle-1 CTS. This
now becomes consistent with single cycle fibs, where we are always using cycle-1 CTS as a reference. This also means
that in essence the only instance where we would not have a CTS reference price would be when we are currently active
on cycle 0."

**Decided on the review point (2026-10-13, verbatim in `qa_findings.md`):** a cross read before its target cycle is
established — "price side only until the break pattern exists" — "Ok".

**Engineering reading.** For a fib record of structure `sid`, direction `sd`, target cycle `T` (`fib_state.cycle_id`
— a single's cycle, a cross's last cycle):

- **`prev_cycle_cts`** applies whenever `T ≥ 1` — a single and a cross alike: cycle `T−1`'s `CTS_CONFIRMED` price
  (`_cts_by_cycle`, the confirmed anchor, never a threshold). Its bound = the first candle of the pattern that
  ESTABLISHED cycle `T` (`_est_pattern_anchor[(sid, T)]`, T's `CTS_ESTABLISHED` `pattern_anchor_idx` — the pattern
  that broke `CTS_{T−1}`), known from T's establishment moment (`_bos_moment_by_cycle[(sid, T)]`).
- **T not established yet** (a pre-established cross, CROSS_CYCLE_FIB_SPEC §4) → the reference has NO bound: the
  price side only (RX1-2a's convention for a reference without a known break).
- **Unchanged:** a single (`a = T` already); the structure reference (`prev_structure_bos` / `cross_limit_anchor`,
  gated by `T` as today); the choice (further opposite; EQUAL → the structure reference, RX1-K5); the condition (price
  side strict, strictly before a known bound); every reader's moment (MS: the fed event's; the drawn POI + the stamp:
  the lifecycle cap).
- **Consequence:** a cross from cycle 0 (`0 → T`) now HAS a `prev_cycle_cts` reference (today: the structure reference
  only, or none) — and the "further opposite" choice may now pick it.

**Supersedes** the cross part of the user's RX1 (2026-10-09, `plan_h_inputs/ic_reference_conditions_spec.md`
paragraphs 4–5 + the recap: "when the cross-fib takes up the entire structure from cycle 0-2, we cannot / do not use
any prev cycle CTS as reference price"; "a valid cross-cycle fib from 1-2, then we can use CTS_0") and PLAN_K §1's
reading of it ("its BOS in cycle `a` … a cross: `meta["cross_start_cycle"]`"). Both keep their text with a dated
pointer here (§5).

---

## 2. Why the reads are always defined (the invariants the code asserts)

- **`CTS_{T−1}` is confirmed for every record with `T ≥ 1`.** A cross with target T exists only from T's
  `pre_established` phase, which starts at `CTS_{T−1}` CONFIRMED (CROSS_CYCLE_FIB_SPEC §4 "Phase machine"; the
  pre-established cross is created on CTS_THRESHOLD_UPDATED of cycle T−1 with T in `pre_established`); a single of T
  exists from T's `CTS_ESTABLISHED`, which the S2 order tripwire refuses before `CTS_{T−1}` CONFIRMED (§1 / PLAN_I F14).
  Kept: `assert cts is not None`.
- **T's pattern anchor and T's establishment moment are written together** (`FibTracker` CTS_ESTABLISHED handler,
  `fib_tracker.py:840` and `:855` at `caf0f01` — `:847` / `:862` after this landing) — both present (T established) or both absent (pre-established). New:
  `assert (anchor is None) == (known is None)`; the two dict reads become `.get` (today's `[...]` raised for no
  reachable case: `a < T` is always established).

**Which readers can see a pre-established cross** (traced; `shadow_change2.py` counts 0 on the window):
- **only the STAMP** (`stamp_ic_references`, every record in `_fibs`, at the lifecycle cap) — a cross still
  pre-established when its structure ends (a sub's end) or at the data edge (main, an open sub: `evaluated_at=None`):
  the fib meta → the fib CSV → the fib hover;
- **never the DRAWN POI:** `select_ic_records` gives no record to a cycle with no CTS_ESTABLISHED in the structure's
  events (`poi_zones.py:820-830`, the zones-audit option N, 2026-09-29) — and the downstream tracker is fed exactly
  those events (a sub's cut at its cap), so `_est_pattern_anchor[(sid, T)]` is missing exactly when the record is
  skipped;
- **never MS's in-flight read:** MS refreshes its POI inner only at CTS_ESTABLISHED / CTS_UPDATED of its current,
  established cycle (`market_structure.py:1492` after the feed, `:1633` after the feed; `representative_fib`'s one
  caller `:1942`) — T is established by then.
(`qa_findings.md` review point (b) said the stamp + MS's read; it was right about the drawn POI, wrong about MS.)

---

## 3. Every reader of the reference (all unchanged in code except the builder + the hover)

| Reader | Where | Moment | What changes |
|---|---|---|---|
| The builder | `FibTracker.ic_reference` (`zones/fib_tracker.py`) | — | `start_cycle` → `target` (§4) |
| The stamp → fib meta `ic_ref_*` → the fib CSV `meta`, the fib hover's IC ref line | `stamp_ic_references` (`orchestrator.py:374`, inside `_run_downstream_pipeline` — the main and every sub projection, `pooled_structure_build.py:91`) | lifecycle cap | the values of every cross record |
| The drawn POI + the IC CSV rows (`ic_ref_reason`, `is_ic_candidate`, the pick) | `select_ic_records` → `select_ic(..., ic_reference=fib_tracker.ic_reference(...))` (`poi_zones.py:836`) | lifecycle cap | crosses' checks; one pick (§6) |
| MS's in-flight POI inner (the sd-proximity CTS confirmation) | `poi_inner_for_fib` (`poi_zones.py:1667`) | the fed event's | 19 M15 reads see another inner; no proximity outcome changes (§6) |
| The fib hover's cycle suffix | `_zone_render.fib_hover_ic_ref_text` | — | reads `cross_start_cycle` today → `cycle_id − 1` (§4) |

Passive readers (no code change): the mirror's index shift of `ic_ref_break_*_idx` (`entity_df_mutation.py:171-172`,
`_FIB_META_IDX_KEYS`); the IC CSV's `ic_ref_reason` column (`debug/export_zones.py:49` / `:110`). `ic_reference` has
exactly the 3 callers above (`fib_tracker.py:514` at `caf0f01` / `:521` after, `poi_zones.py:836`, `:1667` — none in unified_probe / multitf /
charting). `cross_start_cycle` stays (the cross machinery's version transition `fib_tracker.py:1634`, written at
`:1716` — `:1641` / `:1723` after this landing, the exports); only the builder's and the hover's reads of it go.

---

## 4. The code (= `trial_p1_change2.patch` + the assert's message + docstrings)

1. `FibTracker.ic_reference`: drop `start_cycle`; `if target >= 1:` read `_cts_by_cycle[(sid, target − 1)]` (assert
   kept, its message in T's terms naming BOTH guarantees — the S2 order tripwire for an established T, the phase
   machine for a pre-established cross: `pre_established` is set at CTS_{T−1} CONFIRMED, `fib_tracker.py:1296-1299` at
   `caf0f01`, `:1303` after);
   `anchor = _est_pattern_anchor.get((sid, target))`, `known =
   _bos_moment_by_cycle.get((sid, target))`, assert both-or-neither; `brk = (anchor, known)` iff `anchor is not None`
   and (`evaluated_at is None` or `known <= evaluated_at`). The docstring's first bullet: target cycle T, a single and a
   cross alike, the pre-established case (price side only; reachable by the stamp only, §2). Every "TRIAL
   2026-10-13, change 2" wording of the patch → "PLAN_O".
2. `_zone_render.fib_hover_ic_ref_text`: `cycle = f", cycle {int(fib_st.cycle_id) - 1}"` (no TRIAL comment); its
   docstring `[, cycle <T−1>]`.
3. Comments: `zones/cross_cycle_fib.py:147` (`prev_cycle_cts`: "cycle T−1's CTS_CONFIRMED price (the fib's target
   cycle T ≥ 1)").

No event, meta key, CSV column or chart element is added or removed (values only) → no save-format boundary.

---

## 5. Docs (same commit)

- `zones/POI_ZONES_SPEC.md` §3.2: the record's definition (drop "its BOS in cycle `a` … `cross_start_cycle`"), the
  `prev_cycle_cts` bullet in T's terms + the pre-established case, a dated line (2026-10-13, PLAN_O) with this
  landing's values next to the "On the reference window" paragraph (its S_K_A values stay as history).
- `charting/CHARTING_SPEC.md` (the fib hover's IC ref line, ~180): `cycle <T−1>`.
- `GLOSSARY.md` "IC reference": "bound = the pattern that established the fib's TARGET cycle" (was "starting cycle").
- `zones/CROSS_CYCLE_FIB_SPEC.md` §3 "The IC reference's inputs" / "Exported": one line — a cross's `prev_cycle_cts`
  is its target's previous CTS (PLAN_O); §8 (MS's read) unchanged.
- A dated "superseded for crosses by PLAN_O (the user, 2026-10-13)" pointer at the cross sentences — text kept — in:
  `plans/PLAN_K_ic_reference_condition.md` §1 (its later lines `:74 / :225 / :288 / :397 / :421 / :469` are history:
  the §1 pointer covers them); `plans/plan_h_inputs/ic_reference_conditions_spec.md` ¶4–5 + the recap (the user's
  RX1); `plans/PLAN_H_institutional_pattern_poi.md` §8.9 "The rule as DECIDED" (`:1314-1330` — "BOS anchor in cycle
  `a`", "cycle `a`'s CTS_ESTABLISHED `pattern_anchor_idx`"; POI_ZONES_SPEC / GLOSSARY / the register cite it as the
  user's rule); `IDEA_PARKING_LOT.md` the CL-8 line (`:176-183`: sub 3's cross "IC reference is cycle 0's CTS (0.5818,
  bound 2824)") and `plans/PLAN_M_counter_sub_cross_limit.md` `:227-231`, `:250`, `:802-803` (sub 3's reference /
  pick, now CTS_1 0.57902 and 2837).
- Test comments: `test_plan_k_landing_a.py`'s module docstring (the reference bullet) and `:292 / :294 / :301 / :341 /
  :601`; `test_plan_m_landing_1.py:256-257` ("no earlier cycle, so the structure side is its only reference").
- Superseded headers (one line, like `caf0f01`'s on `mutants_n_review.py`): `plans/plan_h_inputs/mutants_k_landing_a.py`
  (F01 / F02 / F04 / F05 / F06 / H2 / H3 target the removed lines; F07 was already BAD at `caf0f01`) and `plans/plan_h_inputs/icref_census.py` (`:203`, the
  `a = cross_start_cycle` rule).
- Register `IDEA_PARKING_LOT.md` §D: the session-order item 2 line → PLAN_O landed (+ PLAN_P pending); memory at
  `/commit-save` (`MEMORY.md` priorities, the session log).
- The `/compare` skill: no format boundary; the chart counts stay (83/239, 137/123, 282/276) if §6 holds.

---

## 6. The numeric prediction (written BEFORE the landing replay; from `shadow_change2.py` on HEAD, 12.1 s)

Exactly the 4 cross records of the window move; nothing in MS does.

| Record (lens) | Today's reference (S_N_hover fib meta) | After | IC CSV: checks passing the reference | Pick |
|---|---|---|---|---|
| H1 sid1 cross 0→1 (BOS 689 → CTS 761, sd −1) | 0.5736 `prev_structure_bos`, break 897 / 902 | **0.57902 `prev_cycle_cts` (CTS_0)**, break 747 / 748 — CTS_0 is higher than 0.5736 | 27 → 19 of 27 | 732, unchanged |
| sub 3 cross 1→2 (BOS 2758 → CTS 3047, sd −1; both lenses) | 0.5818 `prev_cycle_cts` (CTS_0), break 2824 / 2829 | **0.57902 `prev_cycle_cts` (CTS_1)**, break 2991 / 2992 | 26 → 100 of 130 | **2821 → 2837** (12-29 11:15; 1A.iv per `shadow_ic_changes.py` V2) |
| sub 7 cross 0→1 (BOS 4027 → CTS 4157, sd +1; both lenses) | 0.58105 `cross_limit_anchor`, no break | **0.57674 `prev_cycle_cts` (CTS_0)**, break 4154 / 4157 | 61 → 57 of 61 | 4051, unchanged |
| sub 7 cross 0→2 (BOS 4027 → CTS 4222, sd +1; both lenses) | 0.58105 `cross_limit_anchor`, no break | **0.57798 `prev_cycle_cts` (CTS_1)**, break 4178 / 4179 | 90 → 69 of 90 | 4083, unchanged |

- **Singles: 0 changes** (23 of the 27 stamped records; by construction).
- **MS:** 56 H1 reads (7 see another reference, none another inner); 429 M15 reads — 36 see another reference, **19
  another inner**, all sub 3's cross 1→2 between 12-31 02:00 and 12-31 15:45 (inner 0.58261 / 0.58253 / 0.58182 →
  0.58166 / 0.5799). **All 477 sd-proximity checks give the same outcome** (hit, inner and zone kind) under both
  snapshots → **0 structure-event changes** on H1 and both lenses (MS's only use of the inner is that check).
- **Pre-established reads:** 0 (no cross is pre-established at a structure's end on the window).
- **Therefore unchanged:** structure events, KL zones, WVMI (the main gate reads H1 POIs — unchanged; a sub's WVMI has
  no POI gate), triggers, subs, unresolved triggers, the fetch gate (4228 / 5) and the skip-grep (22).
- **Changed CSVs:** the fib lifecycle `meta` of the 4 records (H1 + both lenses); the IC candidates CSV rows of the 4
  records (`ic_ref_reason` / `is_ic_candidate`; sub 3's picked row + its IP columns); the POI zones CSV — sub 3's
  cycle-2 POI, both lenses: IC 2821 → **2837**, zone [0.58182, 0.58233] → **[0.5799, 0.58038]**, `ip_outcome`
  `no_pattern` → `pattern`, `ip_rank` empty → **1A.iv** (+ the pattern meta), `activation_history` one entry 2992 →
  **one entry 3047** (12-31 15:45, the cross's final CTS anchor — the touch as of t holds only once the CTS reaches it).
  Measured by the mechanism reviewer's in-process patched run (no exports): every other POI of H1 and both lenses
  unchanged; lens event counts unchanged (confluence 1346, counter 481).
- **The 3 records whose reference SOURCE switches** (not just its cycle) — the crosses from cycle 0, which had no
  previous-cycle CTS before: H1 sid1 0→1 (`prev_structure_bos` → CTS_0) and sub 7's 0→1 / 0→2 (`cross_limit_anchor` →
  CTS_{T−1}); the "further opposite" choice (RX1-3) is unchanged and now picks the CTS on all three — each stricter
  (checks passing 27 → 19, 61 → 57, 90 → 69), no pick moves. Named for the user's chart look.
- **Charts:** H1 — the sid1 cross 0→1 fib hover's IC ref line (`IC ref 0.57360 (prev_structure_bos) · break 897` →
  `IC ref 0.57902 (prev_cycle_cts, cycle 0) · break 747`); M15 (both lenses) — sub 3's cycle-2 POI rectangle (moves
  to [0.5799, 0.58038], confirm line 2992 → 3047) + its hover (`No pattern · only IC candidate` → `IP 1A.iv · legs
  …`), the IC ref lines of sub 3's and sub 7's cross fibs. **Trace / shape counts unchanged** (83/239, 137/123,
  282/276): one activation before and after, and a pattern POI draws no extra element (`_zone_render.py:184-215`).

---

## 7. Tests

The patched suite (both reviewers, independently): 11 real failures = the 6 tests below (± sd ids); the others are the
environment (`oanda.cfg`, a worktree's MAX_PATH).

Flip — the start-cycle rule's pins:
- `test_plan_k_landing_a.py::test_a_fib_from_cycle_a_takes_cts_a_minus_1_and_cycle_a_s_establishment_pattern` → the
  cross 1→2 takes CTS_1 and cycle 2's pattern (58 / 60); renamed to the target-cycle rule. **Plus a cross 0→2**
  (`_probe(2, sd, start=0)`, `NO_P` → `(1.35, prev_cycle_cts, 58, 60)`): 0→1 and 1→2 have `start == T−1`, so without
  it `cross_start_cycle` in place of `T − 1` survives (review mutant X1).
- `…::test_a_pre_established_cross_reads_only_its_start_cycle` → a cross 1→2 with cycle 1 CONFIRMED (`_conf(45, 1.35,
  1, sd, 40)`) and cycle 2 not established reads `ICReference(_m(1.35, sd), "prev_cycle_cts")` — price side only;
  renamed. (Today's fixture leaves CTS_1 unconfirmed — unreachable for a cross with T = 2, §2; it now trips the assert.)
- `…::test_both_references_take_the_further_opposite_and_equal_takes_the_structure` → the anchors RE-BASED on CTS_1
  1.35 (1.30 → structure, 1.40 → cts, 1.35 → EQUAL → structure) — mandatory: un-rebased, the 1.10 / 1.15 cases pass
  vacuously (1.15 is no longer EQUAL).
- `…::test_the_structure_reference_applies_while_t_is_at_most_m` (`:343`) → the cross 1→2 (T > M, structure side
  dropped) takes CTS_1 `(1.35, 58, 60)`; its `:341` comment reworded.
- `…::test_the_hover_names_the_cross_start_cycles_previous_cts` → "cycle 1" for a cross 1→2 **and "cycle 1" for a
  cross 0→2** (`cross_start_cycle: 0, cycle_id=2` — kills the hover printing `cross_start_cycle`, X2); renamed.
- `test_plan_m_landing_1.py::test_the_structure_reference_takes_the_fixed_answer_else_the_live_one` (`:260`) → a
  cross 0→1 probe of the pre-established target (the tracker built through `on_cts_threshold_updated`) at moment 30 reads
  `ICReference(_m(1.2, sd), "prev_cycle_cts")` — price side only — instead of `ICReference.NONE`; at 31 still the
  `cross_limit_anchor` 1.1 (further opposite); the `:256-257` comment reworded. This is the pin for "a cross from
  cycle 0 gains `prev_cycle_cts`" and for "price side only at a non-None moment". (Built: it keeps its `_probe` — with
  `P` 1.1 at the update 30 the limit allows no cross (CTS_0 1.2 is beyond it), so there is no real record to read; the
  REAL pre-established cross is pinned by the new stamp test below.)

New:
- the both-or-neither assert — BOTH halves with `pytest.raises(AssertionError)` (anchor without moment; moment without
  anchor — the silent half: without the assert it returns a price-only reference; with the assert deleted the first
  half raises a `TypeError`, so `pytest.raises(Exception)` would not kill the deletion);
- a pre-established cross at the data edge THROUGH THE REAL STAMP: `_pre_est(_live(9.0, [], sd), sd, updates=(30,
  31))` then `stamp_ic_references(df, None)` → the cross 0→1's meta `ic_ref_price` 1.2 (sd +1; 1.8 for sd −1 per the
  fixture) / `ic_ref_source` `prev_cycle_cts`, no `ic_ref_break_*` keys.

Mutants (`mutant_runner.py`, the full suite each, `plans/qa_ic_crossfib_inputs/mutants_plan_o.py`; every one KILLED,
on the landed code): `target - 1` → `target`; `target >= 1` → `target > 1`; the anchor / moment keyed by `target - 1`;
drop `anchor is not None`; `known <=` → `known <`; `.get` → `[...]`; `brk`'s tuple swapped; crosses from cycle 0
excluded; the CTS keyed by `cross_start_cycle` for a cross (X1); the builder fully back to `cross_start_cycle` (the
HEAD code); the both-or-neither assert deleted / neutralized; the hover's `- 1` dropped; the hover printing
`cross_start_cycle` (X2); the hover back to `cross_start_cycle - 1`. Equivalent (expected to survive, not a gap):
`anchor is not None` → `known is not None` (the assert makes them one).

---

## 8. The landing

1. The code (§4) + tests (§7) + docs (§5); full suite from the repo root (the OANDA smoke test can fail transiently).
2. Replay (`python -m engine_v2.run_replay > run.log 2>&1`): fetch gate, skip-grep, timing.
3. `/compare` vs S_N_hover (`cmp_save.py` + `cmp_hover.py`): every delta checked against §6 — expected vs unexpected.
4. Landing review (multi-agent, read-only; size + tokens stated first); mutants.
5. The user's chart look (H1 first) → commit on the user's go → `/commit-save` (the new `LATEST_<branch>`; PLAN_P's
   baseline).

## 9. Review log

**Plan review, 2026-10-13** — 2 read-only lenses (a mechanism / readers / prediction lens: 217,421 tokens, 58 tool
calls, re-ran the shadow + a patched in-process run; a tests / docs / mutants lens in a worktree reset to `caf0f01`:
169,655 tokens, 55 tool calls, the patched suite + scratch mutants) = **387,076 tokens**. No BLOCKER.

| # | Severity | Finding | Disposition |
|---|---|---|---|
| R1 | MAJOR (both lenses) | §7 missed 2 failing tests (4 ids): `test_the_structure_reference_applies_while_t_is_at_most_m`, `test_plan_m_landing_1::test_the_structure_reference_takes_the_fixed_answer_else_the_live_one` | §7 Flip, the plan_m one as the real pre-established pin |
| R2 | MAJOR | Every cross in §7 had `start == T−1` (0→1, 1→2): X1 (the CTS by `cross_start_cycle`) and X2 (the hover printing `cross_start_cycle`) SURVIVED | a cross 0→2 in the builder and the hover tests; X1 / X2 in the mutant list |
| R3 | MINOR | The EQUAL case passes vacuously unless re-based | re-basing mandatory (§7) |
| R4 | MINOR | The both-or-neither pin: both halves, `AssertionError` | §7 New |
| R5 | MINOR | No pin through the real stamp for a cross | §7 New (`_pre_est` + `stamp_ic_references`) |
| R6 | MINOR | §2 overcorrected: the drawn POI never sees a pre-established target (`poi_zones.py:820-830`) — only the stamp | §2 rewritten; verified at the code; `qa_findings.md` corrected |
| R7 | MINOR | §6's one unpredicted figure measurable: 2837's activation = one entry 3047; zone [0.5799, 0.58038]; 1A.iv; counts hold | §6 |
| R8 | MINOR / NIT | §5 missed PLAN_H §8.9, the register CL-8 line, PLAN_M sub-3 lines, test comments, two stale tools | §5 |
| R9 | NIT | §3 left out 2 passive readers (the mirror's shift, the CSV column) | §3 |
| R10 | NIT | TRIAL wording; the assert message names only the S2 tripwire | §4 |
| R11 | NIT | Tell the user 3 of 4 moves switch the reference's SOURCE (crosses from cycle 0) | §6 + the chart look |

Confirmed by the lenses (unchanged): §2's invariants (one writer each, `:840` / `:855`, one handler; MS snapshots the
whole tracker; a second establishment raises), MS never reads a pre-established target, the 3 callers, singles
unchanged, `ref_v2` == the patch, the [MSPROX] proof (`poi_inners_for_cycle` read only in the proximity check), every
§6 STATS figure (re-run, 14.4 s), §4 covers every production site of the old rule, no skill describes it.

## 10. The landing, as measured (2026-10-13, uncommitted on `caf0f01`)

- **Suite:** 1718 passed (1712 + 6 new: `test_plan_o_ic_reference_target_cycle.py`), 81 s.
- **Replay:** fetch gate PASS (4228 / 5), silent-skip grep 22 (= S_N_hover: the 4 unresolved collapsed-parent
  triggers + their pool / sweep lines, the 2 collapsed parent cycles, the parent-tables lines, the fib-render line, 5
  pandas FutureWarnings), timing 73.5 s (multi_tf_dual 11.9, charts 11.6 / 13.8 / 32.2 — machine variance).
- **`/compare` vs S_N_hover** (`cmp_save.py` + `cmp_hover.py` + a per-record keyed check): exactly 8 of 27 CSVs
  differ, every cell as §6 — the 3 fib-lifecycle CSVs (the 4 cross records' `ic_ref_*`: H1 0.57902 / 747 / 748; sub 3
  0.57902 / 2991 / 2992; sub 7 0.57674 / 4154 / 4157 and 0.57798 / 4178 / 4179), the 3 IC CSVs (passing the reference
  27 → 19, 26 → 100, 61 → 57, 90 → 69; IC candidates H1 12 → 11, sub 3 1 → 55, sub 7 unchanged 21 / 44; picks only sub
  3 2821 → 2837), the 2 lens POI CSVs (sub 3: 2837, [0.5799, 0.58038], 1A.iv legs 3-1-3 2834–2836 | 2837 | 2838–2840,
  confirmed / activation 3047). Structure events, KL, WVMI, subs, triggers, unresolved, imbalances, final / raw /
  levels: md5-equal. Chart counts 83/239, 137/123, 282/276 (unchanged).
- **Figures:** H1 IDENTICAL (traces x/y, shapes, hovertemplate, customdata). Each lens: 5 traces' customdata (sub 3's
  2 POI-zone hover traces — 769 → 753 points, the zone starts 4 hours later — and the 3 cross fibs' hover: sub 3's
  `No pattern · IC 2821 · only IC candidate | IC ref 0.58180 (prev_cycle_cts, cycle 0) · break 2824` → `IP 1A.iv · IC
  2837 | IC ref 0.57902 (prev_cycle_cts, cycle 1) · break 2991`; sub 7's `IC ref 0.58105 (cross_limit_anchor) · break:
  none` → `IC ref 0.57674 (prev_cycle_cts, cycle 0) · break 4154` / `0.57798 (…, cycle 1) · break 4178`) + 3 shapes:
  sub 3's two POI rectangles (the one from the IC 12-29 07:15 → 11:15, the one from the activation 12-31 02:00 →
  15:45) and its confirm line (2992 → 3047).
- **Prediction wrong in detail (one line of §6):** the H1 fib hover of sid 1's cross 0→1 does NOT change — that fib
  is on the collapsed parent cycle (1,1) and is not drawn on H1 (`[chart][fib] rendered 3/5`); its new reference is
  in the CSVs only.
- **Mutants** (`qa_ic_crossfib_inputs/mutants_plan_o.py`, 17, full suite each, 4 min 49 s in parallel): 16 KILLED;
  O16 (`anchor is not None` → `known is not None`) SURVIVED as predicted — equivalent under the both-or-neither
  assert. X1 / X2 are killed only by the new 0→2 cases (1 / 2 failing tests).
- **Landing review (2026-10-13)** — 1 read-only lens, 218,728 tokens, 65 tool calls: no BLOCKER / MAJOR; the code =
  §4, the tests pin §7 (every flipped / new assertion fails under HEAD's builder; the EQUAL case not vacuous), every §10
  number re-measured against the artifacts (md5 8 / 27; the H1 html identical once the plot-div UUIDs are stripped).
  Its 8 findings, all docs, fixed before the commit: the register's item-2 line (PLAN_O landed, PLAN_P next — LR1);
  two more PLAN_M sentences of the old rule (LR2); the `/compare` skill's PLAN_O behaviour-change entry — added at
  `/commit-save` with the save name (LR3); the stale-mutant list (+ F05 / F06; F07 was already BAD — LR4); the census
  header's direction (LR5); line refs marked `caf0f01` (LR6); §7's plan_m bullet (LR7); §10's 3 shapes (LR8).
- **The user's chart look (2026-10-13):** "Charts look good, go ahead and commit once the review is done".
