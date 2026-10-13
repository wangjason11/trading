# Plan M — The counter-sub cross limit: the parent's BOS threshold, read live (Week 9 Part 1, register §D.1 item 2)

**Status (2026-10-11, session 1): RULE + DESIGN DECISIONS DONE (rounds 1–2, §3); the census measured (§6); the
design + stages (§7–§10) COLD-REVIEWED (760,821 tok, 0 BLOCKER on the design, every finding folded — §11); the
export's "final" decided (a) = as of the structure's end (§3); LANDING 1 BUILT + LANDING-REVIEWED (§12, §13: byte-identical
vs S_L2, the census unchanged, suite 1657, mutants 25 / 26 + 1 equivalent + the review's 4 / 4; review 451,068 tok,
0 BLOCKER / MAJOR, every finding folded); the user's go "sounds good go ahead" → COMMITTED `86d4f7b`, SAVED
S_M1 `20261011_204047_86d4f7b` (save `8bdcc8a`, trunk `a442e91`) = `LATEST_<branch>`; LANDING 2 BUILT + LANDING-REVIEWED (§14–§15:
= the trial's outputs exactly, the census PINNED table 14 / 14, 0 knock-on, suite 1691, mutants 25 / 25 + the
review's 6 / 6; review 437,943 tok, 0 BLOCKER / MAJOR, all folded); the user's chart review (2026-10-12): "charts look
good. go ahead" → COMMITTED `10e4c6f` + SAVED S_M2 `20261012_023137_10e4c6f` (§16). PLAN_M COMPLETE; the parked
follow-up CL-8 CLOSED 2026-10-12 with no change (the user: "keep it", §17).** Living document:
every question, answer and decision is recorded here as it happens. Register line: `IDEA_PARKING_LOT.md` §D.1 item 2
(the cross-limit price debugging, opened by the user 2026-10-09).

**Input (the user's write-up, VERBATIM, canonical):** `plan_m_inputs/counter_sub_cross_limit_writeup.md` (2026-10-10).
**Amends:** PLAN_I's rule that `P` is fixed once, at the structure's start (PLAN_I §1.2 / §1.4, decision R2-4), for the
subs this plan names; CROSS_CYCLE_FIB_SPEC §3 (the limit, its sources, the IC reference's inputs) changes with the code.
**Baseline:** S_L2 `artifacts/commits/week9-entries-decisions/20261010_155816_6d8cc67` (source `6d8cc67`, save
`8aa4f80`) — the design's; since landing 1 **S_M1** `20261011_204047_86d4f7b` (source `86d4f7b`, save `8bdcc8a`,
trunk `a442e91`; byte-identical to S_L2, §12); since landing 2 **S_M2** `20261012_023137_10e4c6f` (source `10e4c6f`,
save `3ebe857`, trunk `475bbf9`; = `LATEST_<branch>`, §16).
**Tools (`plan_m_inputs/`, read-only):** `limit_table.py <save>` (every structure's `P` / source / `M` / fib records /
IC reference), `struct_events.py <save> H1:<sid> sub:<id> [--all-updates]` (CTS / BOS / state events with times),
`trial_diff.py BASE CUR_DEBUG [CUR_CHARTS]` (a keyed diff of a replay vs a save), `fig_hash.py CHARTS_DIR` (full-figure
fingerprints), `census_patch.py` / `census_run.py` / `census_report.py` (the live-`P` census on other windows, §6),
`cl8_candles.py` (the CL-8 case's M15 / H1 candles off S_M2, §17).
**The trial's outputs, persisted** (§4, §8): `trial_md5.txt`, `trial_fig_hash.txt`, `trial_csvs/` (the 6 changed CSVs).

---

## 1. The rule (the engineering reading of the write-up + round 1)

1. **Which subs — a COUNTER sub:** a unique sub whose direction is opposite its FIRST record's parent (the record's
   `relative_dir == "counter"`, set by the sweep from the trigger's direction vs the parent sid's), whatever its
   trigger type (`reversal`, `first_counter`, `subsequent_counter`) and whatever its lens (§3 CL-4). On the window:
   subs 1, 3, 5, 7. Every other sub — and every H1 sid — keeps its rule unchanged (the user: "We don't change
   anything for any other triggers"). **"First record" = the CREATING record at the fire** — the limit is computed
   when the sub's geometry is built, at its first trigger's fire (PLAN_I R6-2 (a), as for every source); the R6-2
   tripwire re-derives it from the started_by record (the first LIVE one by start) and WARNs on a mismatch, the
   stored `P` staying (§7.3 names the two shapes). **The rule, not every value:** a counter sub's own structure can
   still change (its MS reads `P` through the sd-proximity POI inner), and with it the `P` of a sub that reads it
   as x−1, or a reversal of it — 0 on the window (§4); the post-landing census lists such changes separately (§9).
2. **Its `P`:** the parent's BOS threshold of the first record's parent sid `S` and cycle `C` — the parent's
   `BOS_CONFIRMED(S, C)` price, moved by every `BOS_THRESHOLD_UPDATED(S, C)` (MS moves it only FURTHER: a close-break
   whose reversal anchor fails at once, or a wick past it without a close → that candle's extreme;
   `MarketStructure._bos_barrier_step`). Read LIVE: an H1 threshold event of moment `h` is known on the sub's frame at
   the last M15 candle of H1 hour `h` (the timing mapper, `_map_parent_idx_to_m15_hour_end`). Frozen at the parent
   cycle's end — its next cycle's start or the parent's reversal (§3 CL-2; this also covers a counter sub that
   continues under the NEXT sid after the parent's reversal). So `P` only ever moves further. (A move by a failed
   close-break — `rv_anchor_failed` — carries MS's own moment, the break candle, although MS decided there was no
   reversal pattern by reading up to 5 H1 candles ahead (`_schedule_reversal_from_anchor`): MS's convention for
   its own threshold, inherited as is.)
3. **When a moved `P` counts (§3 CL-1):** a cycle T ≥ 1 reads `P` (and so `M`) as of the moment its cross permission
   is checked, until T is ESTABLISHED; from its establishment T's permission is fixed for its life. Every read before
   the sub's start uses `P` as of the start (the unique sub is built whole at its start, as today). Since `P` only
   moves further, a moved `P` only gives LATER cycles more room — the user: "the limit increases meaning we need a new
   1st CTS to be past it".
4. **The IC reference (§3 CL-3):** its structure side follows the new `P` — the anchor behind it is the parent's
   `BOS_CONFIRMED(S, C)` price; its bound = the break of the parent's threshold, found by the EXISTING M15
   cross-limit breakout watch on `P` as of the reading moment (CL-7; `ic_ref_source` `cross_limit_anchor`, the
   label of every non-reversal source) — NOT the parent's realised H1 reversal pattern that the retroactive sid 1
   uses (`prev_structure_bos`). On the window the two agree (the watch's break anchor 3588 = H1 897's first M15
   candle); in general the M15 watch can see a break inside an hour that H1 never treats as one — and when H1 then
   moves its threshold past it, that break stops counting (CL-7).
5. **Unchanged:** `beyond` (strict), `M` = the first cycle whose latest CTS anchor is beyond `P` = the last cycle a
   cross may target, the dead-cycle walk, the own-imbalance test, the version transitions, `cross_failed`, cycle 0
   always single, no single + cross coexistence (CROSS_CYCLE_FIB_SPEC §2–§5).

**Why (the write-up):** the retroactive H1 structure a reversal writes over the past (sid 1, `P` = sid 0's frozen BOS
threshold 0.5736, `M` = 2) is what the live counter sub (sub 3) mimicked at the same BOS / CTS prices — but sub 3
obeyed its own reversal limit (sub 2's frozen BOS 0.58156), much nearer. Taking the parent's threshold gives the live
sub the limit the retroactive structure will have. It is close to "limitless" in practice: the parent's break of that
threshold is its reversal, which ends the sub's record (`parent_end`) — the exceptions are the LOH lag (an H1 move is
known at its hour's last M15 candle, so inside that hour a sub's CTS can sit past the not-yet-moved threshold — §6's
recurring shape), the hours a reversal pattern is pending (no threshold update while MS's reversal watch is open),
and false breaks (which move the threshold further).

## 2. Measured — the status quo on S_L2 (session 1; `plan_m_inputs/limit_table.py`, `struct_events.py`)

Times UTC; M15 index i = H1 candle i // 4 + 15 min × (i % 4); 1 pip = 0.0001.

**Every structure's limit today:**

| Structure | Dir | Started by (start) | `P` | Source (from) | `M` | Fibs |
|---|---|---|---|---|---|---|
| H1 sid 0 | buy | start trading | none | none | – | 2 singles |
| H1 sid 1 | sell | reversal (902, 01-09 12:00) | 0.5736 | sid 0's reversal frozen BOS | 2 | cyc 0 single, cyc 1 cross from 0, cyc 2 single (`cross_failed`) |
| sub 0 | buy | first_confluence, parent cyc 0 (1020) | none | none | – | 5 singles |
| sub 1 | sell | reversal (1940, 12-15 03:00) | 0.57877 | sub 0's reversal frozen BOS | 2 | 5 singles |
| sub 2 | buy | reversal (2470, 12-22 15:30) | 0.57926 | sub 1's reversal frozen BOS | 0 | 2 singles |
| sub 3 | sell | reversal (2829, 12-29 09:15) | 0.58156 | sub 2's reversal frozen BOS | 1 | 4 singles |
| sub 4 | sell | first_confluence, parent sid 1 cyc 2 (3621) | 0.57376 | parent's cycle-1 CTS threshold | 0 | 1 single |
| sub 5 | buy | first_counter (3759, 01-13 01:45) | 0.58105 | sub 4's BOS threshold | none | 1 single |
| sub 6 | sell | subsequent_confluence (3819) | 0.57112 | sub 5's BOS threshold | none | 1 single |
| sub 7 | buy | subsequent_counter (4159, 01-19 05:45) | 0.57827 | sub 6's BOS threshold | 2 | cyc 0 single, cyc 1–2 crosses from 0 |

**Sub 3 vs the retroactive H1 sid 1 — the same BOS / CTS prices, shifted one cycle** (sub 3 has an extra cycle 0):

| sub 3 (M15, live 12-29 09:15 → 01-09 12:45, `parent_end`) | H1 sid 1 (retroactive; starts 902) |
|---|---|
| cyc 0: BOS 2639 (12-24 09:45) 0.58478 → CTS 2736 (12-26 10:00) 0.5818 | — |
| cyc 1: BOS 2758 (12-26 15:30) 0.58424 → CTS 2842 (12-29 12:30) 0.57902 | cyc 0: BOS 689 (12-26 15:00) 0.58424 → CTS 710 (12-29 12:00) 0.57902 |
| cyc 2: BOS 2915 (12-30 06:45) 0.58196 → CTS 3047 (12-31 15:45) 0.574 | cyc 1: BOS 728 (12-30 06:00) 0.58196 → CTS 761 (12-31 15:00) 0.574 |
| cyc 3: BOS 3304 (01-06 08:00) 0.58105 → CTS 3611 (its end) 0.57228 | cyc 2: BOS 826 (01-06 08:00) 0.58105 → CTS 905 (01-09 15:00) 0.57112 |

- **A correction to the write-up (agreed in round 1):** sub 3's cycle-0 CTS 0.5818 is NOT past 0.58156 (2.4 pips
  short); cycle 1 (0.57902) is the first past it → `M` = 1. Cycle 1 was allowed a cross, but fell back to a single
  (`cross_failed` — no live earlier cycle at that moment); cycles 2–3 are singles past the limit. Net: no cross fib on
  sub 3, as the user saw; the new rule does not touch cycle 1.
- **The parent's threshold never moves on this window:** 0 H1 `BOS_THRESHOLD_UPDATED` events (every sid). sid 0's
  cycle-1 threshold sits at 0.5736 from 652 (12-24 02:00) to the reversal (pattern 897 → apply 902) — so sub 3's
  live `P` would equal sid 1's exactly, and the live mechanics (§1 items 2–3) are NOT exercised by a replay of this
  window (tests + a random-slice corpus must cover them).
- **"Sub 3 ends anyway" — in the code:** a trigger record ends at its parent CYCLE's end — the next parent cycle's
  start or the parent's reversal (`lifecycle_sweep`: `tables.end(S, C)`, reason `parent_end`) — at the hour's last
  M15 candle: sub 3 ended at 3611 = H1 902.
- **The exception, measured:** sub 3's cycle 3 went past 0.5736 at 3589 (01-09 07:15, 0.57318) — inside H1 897
  itself (07:00–08:00), before H1 897's close-break was known at its last M15 candle 3591 (the LOH lag), and sid 0's
  reversal pattern then stayed pending (897 → 902) — 5.5 hours before the sub ended. "Crosses up to the first cycle past the
  limit" absorbs it: `M` = 3, cycle 3 stays eligible; only a cycle 4 born in those hours would be single.

**Read off the save (no replay) — what the rule changes on the window** (the counter subs; "eligible" ≠ drawn — the
dead-cycle walk still decides each cross):

| Sub | Today `P` · `M` | New `P` (parent cycle's BOS) · `M` | Effect |
|---|---|---|---|
| 1 sell, sid 0 cyc 0 | 0.57877 · 2 | 0.55808 (BOS 96, 11-20 22:00) · none | cycles 3–4 become cross-eligible |
| 3 sell, sid 0 cyc 1 | 0.58156 · 1 | 0.5736 (BOS 591, 12-19 13:00) · 3 | cycles 2–3 become cross-eligible (as sid 1's) |
| 5 buy, sid 1 cyc 2 | 0.58105 · none | 0.58105 (BOS 826, 01-06 08:00) · none | the same candle and price — the source label only |
| 7 buy, sid 1 cyc 2 | 0.57827 · 2 | 0.58105 · none (its highest anchor 0.58034) | crosses unchanged; the IC reference 0.57806 + break 4188 → 0.58105, likely no break |

The IC reference also moves on sub 1 cycles 0 / 2 and sub 3 cycles 0 / 1 (cycle 0's becomes the far parent BOS —
effectively no constraint, as sid 1's cycle 0 has today). The trial replay (§4) measures the cascade.

## 3. Decisions

### Round 1 (2026-10-10, session 1) — the user's answers "1. your rec is good 2. b 3. yes 4. yes"

- **CL-0 — the rule (the write-up):** "for all subordinate structure, or specifically triggers … in the opposite
  direction of the first record parent (regardless of the sub trigger), their cross limit will simply be the
  bos_threshold (active) of the parent sid & cycle … We don't change anything for any other triggers." Preferred over
  "limitless": no `P` means NO cross today (sid 0, sub 0), so a real `P` keeps one meaning; the IC reference stays
  tied to the anchor behind `P` (PLAN_K) and becomes the parent's BOS — the one sid 1 uses.
- **CL-1 — when a moved `P` counts:** "your rec is good" — each cycle reads `P` as of the moment its permission is
  checked until it is established; from its establishment the permission is fixed for its life (no single ↔ cross
  swap mid-cycle); every read before the sub's start uses `P` as of the start. (The alternative, re-reading at every
  update, would let a single turn into a cross mid-cycle — a new transition.)
- **CL-2 — a counter sub that outlives its first record's parent cycle** (only through a later record on the
  parent's next cycle; 0 on the window): **(b)** keep the first record's parent cycle's threshold, frozen at that
  cycle's end. (Not (a), following the parent's current threshold, which can move nearer.) Consequence: `P` only
  ever moves further.
- **CL-3 — the IC reference:** "yes" — its structure side follows the new `P`: the parent's `BOS_CONFIRMED(S, C)`
  price, bounded by the break of the parent's threshold.
- **CL-4 — "opposite direction":** "yes" — the sub's direction vs its FIRST record's parent, not the lens (sub 3's
  first record sits in the confluence lens — a reversal of sub 2 — but it is a sell sub under a buy parent).

### Round 2 (2026-10-11, session 1) — the user's answer "ok we can go with your recommendations" (the questions: §5)

- **CL-5 — the source's name:** `parent_bos_threshold` (pairs with `parent_cts_threshold`; a persisted label — the
  fib meta, the CSVs, the hover).
- **CL-6 — the export when `P` moved:** (a) `cross_fib_limit_price` stays ONE value per structure — the FINAL `P`;
  two keys appear only when it moved: the `P` at the sub's start and the number of moves; the hover adds
  `(moved N× from <start P>)`. Why: the smallest export; a move's effect on a cycle already shows in its `via`.
- **CL-7 — the IC reference's bound when `P` moved:** (a) the breakout watch reads `P` as of each moment it checks, so
  a break of a `P` the parent later moved past stops counting. Why: the H1's own false break — the threshold moved,
  so it was not a break.
- **CL-8 — the residual IC difference** (sub 3 cycle 2's IC 2821 vs sid 1 cycle 1's 732, §4): (a) PARKED as its own
  register item, looked at after this plan lands (one cause per `/compare`) — `IDEA_PARKING_LOT.md` §D.1 item 2's
  follow-up line. **Looked at 2026-10-12 → CLOSED, no change** (the user: "1, keep it — close CL-8 with no change";
  §17).

### Readings the cold review made explicit (2026-10-11 — shown to the user; §11)

The user (2026-10-11), on the first one — the question "(a) final as of the sub's end (recommended) / (b) final as of
the parent cycle's end", the 2025-08-01 sub 3 case: **"a, go ahead with landing 1"** — CL-6's reading DECIDED (a).

- **CL-6's "final `P`" = `P` as of the STRUCTURE'S END** (its lifecycle cap; an open sub → the last move), never a
  move after the sub ended — the IC reference's "final" convention (`stamp_ic_references`, PLAN_K §11.2 LA1). The
  case (census, 2025-08-01 → 09-20, sub 3, life 3247 → 3259, `same_dir_replacement`): its parent cycle's threshold
  moves 13 times, all AFTER the sub ended; the export stays 0.59149 with no move keys — not 0.58538 / "moved 13×", a
  `P` the sub never saw. The limit OBJECT still carries every move up to the parent cycle's end (it is built at the
  fire, before the sub's end is known); only the export reads it as of the end.
- **The IC reference's T ≤ M gate is fixed at establishment too** — CL-1 carried through PLAN_K's rule that the
  structure side applies exactly where a cross is allowed (`_structure_reference_limit` = `_cross_allowed`'s test);
  not a separate decision.
- **The break detector (CL-3 / CL-7):** the existing M15 cross-limit watch (`cross_limit_anchor`) — §1 item 4.
- **CL-7's other side (landing 1's review, A NOTE-7 — derived, shown to the user):** the watch at moment `t` reads
  `P(t)` over the whole frame from the BOS_0 anchor (PLAN_K K6), so from a move's moment the EARLIEST break of the
  NEW `P` counts — even one that confirmed BEFORE the move was known. Reachable only when an M15 close went past the
  future threshold before H1 moved there (while MS's reversal watch held the threshold, or inside the LOH lag);
  traced, not measured — the L2 census counts it (§9).

## 4. The trial replay (session 1, 2026-10-10) — the window's prediction

**What ran:** `plan_m_inputs/counter_limit_trial_patch.py` on a `git archive` of `27ab6da` (= S_L2's engine): a counter
sub's `P` = the parent's `BOS_CONFIRMED(S, C)` price, source `parent_bos_threshold`, anchor = that price (so the IC
reference's `cross_limit_anchor` + the breakout watch); the patch ASSERTS the parent cycle has no
`BOS_THRESHOLD_UPDATED` — the window has none, so the trial = the decided rule on this window (no move → the live
reading, the freeze at establishment and the steps are all no-ops). Replay in `%TEMP%/pm/w1`, 47 s wall; M15 fetch
4228 candles (= S_L2's). The `[trial]` log: exactly subs 1 (P 0.55808), 3 (0.5736), 5 (0.58105), 7 (0.58105) at
their fires; the post-sweep re-derivation check 3 subs, 0 mismatches. Outputs: `plan_m_inputs/trial_diff_vs_S_L2.txt`
(`trial_diff.py`), `limit_table_trial.txt` vs `limit_table_S_L2.txt`.

**Unchanged (md5):** every H1 CSV, both lenses' structure events, KL zones, WVMI, triggers, unresolved triggers and
subs CSVs — no structure / trigger / sub change, so MS's sd-proximity reads changed no answer.

**Changed — 6 CSVs (the fib lifecycle, IC candidates and POI zones of both M15 lenses):**

| Sub | Fib records | IC reference | POI |
|---|---|---|---|
| 1 | cyc 3, 4: `via` `cross_fib_limit` → `cross_failed` (now allowed; no live earlier cycle) | cyc 0: 0.57882 `prev_structure_bos` (break 1936) → 0.55808 `cross_limit_anchor`, no break; cyc 2: 0.57882 → 0.57877 `prev_cycle_cts` | none |
| 3 | **cyc 2: single (BOS 2915 → CTS 3047) → CROSS v0 from cyc 1 (BOS 2758 → CTS 3047)**, pre-established at 2967 (12-30 19:45); cyc 1's single ends 2967 instead of 2992 (Option A); cyc 3: `cross_fib_limit` → `cross_failed` | cyc 0: 0.58198 (break 2824) → 0.5736 `cross_limit_anchor`, break 3588 (confirmed 3589); cyc 1: 0.58198 → 0.5818 `prev_cycle_cts` | **cyc 2: IC 2969 (12-30 20:15, pattern 3C.iv, zone 0.57926–0.57944, confirmed 3044) → IC 2821 (12-29 07:15, no pattern — the one candidate, zone 0.58182–0.58233, confirmed 2992)**, both lenses |
| 5 | the limit's source / from-sub only (`prev_sub_bos_threshold` from sub 4 → `parent_bos_threshold`; the same 0.58105) | unchanged | none |
| 7 | the limit (0.57827, `M` 2 → 0.58105, `M` none) | cyc 0–2: 0.57806 + break 4188 → 0.58105, no break (21 IC-check rows change reason on cyc 2; pick 4083 kept) | none |

Every record of the four subs: `cross_fib_limit_price` / `_source` / `_cycle` / `_from_sub_id` per the table of §2.
Three relabelled singles carry their meta keys in another order (the `cross_failed` activation path). Figures: trace /
shape COUNTS unchanged (H1 83 / 239, confluence 282 / 276, counter 137 / 123); the changed traces are the hovers of
the changed records (confluence 8 M15 fib + 2 M15 POI zone traces, 8 shapes; counter 6 + 2).

**Sub 3 now replays sid 1:** the cross on its middle cycle and `cross_failed` on its last — sid 1's cycle-1 cross
from cycle 0 (BOS 689 → CTS 761) is pre-established at 741 (12-30 19:00) on the same CTS-threshold move (0.57899)
as sub 3's at 2967 (12-30 19:45); same fib anchors (0.58424 → 0.574), so the same 61.8–80 zone.

**What still differs — the IC (a residual of sub 3's extra cycle 0, NOT this plan's rule):** sid 1's cycle-1 POI is
IC 732 (12-30 10:00, pattern 3A.iv, zone 0.58068–0.58118); sub 3's cycle-2 POI is IC 2821. sid 1's cross starts at
its cycle 0, which has no earlier cycle → its IC reference is the far structure side (0.5736, break 897). Sub 3's
cross starts at its cycle 1 (it has one more cycle, 2639 → 2736, before sid 1's first) → the IC reference is cycle 0's
CTS 0.5818 (the higher), bounded by the pattern that established cycle 1 (2824, 12-29 08:00) — so 732's M15 period
(12-30 10:00, after 2824; low 0.58068 < 0.5818) is cut, and only 2821 remains. [Superseded 2026-10-13 by PLAN_O (the
user's change 2: a cross's CTS reference = its TARGET's previous CTS): sub 3's cross reads CTS_1 0.57902 (bound 2991)
→ 2837 1A.iv; sid 1's cross reads CTS_0 0.57902 (bound 747) → 732 kept.]

(Measured: the trial's IC check rows of sub 3's new cycle-2 record — 130 rows, ONE candidate, 2821; the M15 candles
of 12-30 10:00–10:30 (2928–2930) touch the zone and have an unfilled imbalance after, and fail only the IC reference,
`ic_past_reference`; H1 732's low is 0.58068.)

## 5. Round-2 questions (asked 2026-10-10, session 1; ANSWERED 2026-10-11 — the recommendations: §3 CL-5…CL-8)

1. **The new source's name** (a persisted label: the fib meta / CSV / hover): `parent_bos_threshold` (pairs with
   `parent_cts_threshold`; the trial used it) — or another.
2. **What the export shows when `P` moved** (never on this window): (a) `cross_fib_limit_price` = the FINAL `P` (as
   today: one value per structure) + two keys only when it moved — the `P` at the sub's start and the number of moves;
   the hover adds `(moved N× from <start P>)`; (b) per record, the `P` its own cycle's decision used (as of its
   establishment) next to the final; (c) the full list of moves (moments + prices) in the meta. Recommended (a): the
   smallest; a move's effect on a cycle is visible in its `via`.
3. **The IC reference's bound when `P` moved:** (a) the breakout watch reads `P` as of the reading moment — a break of
   a `P` the parent later moved past stops counting (as the H1's own false break: the threshold moved, so it was not a
   break); (b) a confirmed break of any earlier `P` keeps counting. Recommended (a).
4. **The residual IC difference** (§4: sub 3 cyc 2's IC 2821 vs sid 1 cyc 1's 732 — sub 3's extra cycle 0 makes its
   cross start one cycle later in its own numbering, so its IC reference is cycle 0's CTS): (a) park it as a new
   register item, looked at after this plan lands (one cause per `/compare`); (b) fold it into this plan.
   Recommended (a). [Superseded 2026-10-13 by PLAN_O (a cross's CTS reference = its TARGET's previous CTS): sub 3's
   cross reads CTS_1 → 2837; sid 1's reads CTS_0 → 732.]

## 6. The live-`P` census (session 1, 2026-10-11) — the parent's threshold DOES move on real data

**What ran:** `plan_m_inputs/census_patch.py` (prints only — the engine's outputs unchanged) on a `git archive` of
`27ab6da`, `census_run.py` (the pipeline without exports, 5–13 s a window, all exit 0) on the reference window + five
other ~7-week NZD_USD windows; `census_report.py` replays the DECIDED rule offline on every counter sub — its cycles'
establishment moments + confirmed CTS prices, the parent cycle's BOS + threshold moves at their LOH moments. Output:
`plan_m_inputs/census_pre_build.txt` (window indices are each window's own frame).

| Window (config start → end) | H1 cycles | H1 threshold moves | Counter subs | … with a move inside their life |
|---|---|---|---|---|
| 2025-12-01 → 2026-01-20 (the reference) | 5 | 0 | 4 (subs 1, 3, 5, 7) | 0 |
| 2025-02-01 → 2025-03-22 | 8 | 9 (on 2 cycles) | 3 | 2 (sub 4: 2 moves; sub 6: 7) |
| 2025-04-01 → 2025-05-20 | 2 | 18 (1 cycle) | 2 | 2 (subs 3, 4: 9 each) |
| 2025-06-01 → 2025-07-20 | 1 | 0 | 1 | 0 |
| 2025-08-01 → 2025-09-20 | 3 | 13 (1 cycle) | 2 | 1 (sub 5: 13) |
| 2025-10-01 → 2025-11-20 | 2 | 0 | 2 | 0 |

36 subs, 14 counter, 5 with a move inside their life; 28 counter-sub cycles ≥ 1 established on or before the sub's
end — 15 inside its life, 13 before its start (read at `P` as of the start, CL-1). (Report re-run after the cold
review: cycles before the start read at `P(start)`, "the end" = `P` as of the sub's end; the numbers below are the
re-run's. The offline model holds each sub's geometry fixed and ignores the IC watch.)

**Facts the design's asserts rest on (every window):** every H1 cycle has exactly ONE `BOS_CONFIRMED` (21 cycles);
every threshold move is strictly further than the threshold before it (40 moves); no move at or after its cycle's end
(`end_h1`) — MS stamps a `BOS_THRESHOLD_UPDATED` with the CURRENT cycle, so CL-2's freeze at the parent cycle's end
holds by construction (the design asserts it rather than filtering); on all 14 counter subs the `BOS_CONFIRMED` is
the first step and known by the sub's start; no two steps share an M15 moment (the collapse rule is synthetic-only).

**The one cycle the live reading changes** (vs `P` fixed at the sub's start) — 2025-02-01 → 03-22, sub 6 (buy;
parent sid 1 cycle 1, a sell; life 2555 → 3307, `parent_end`): `P` at its start = the parent's BOS price 0.57328; its
cycle-3 CTS 0.57597 is beyond it, so with a fixed `P` → `M` = 3 and cycle 4 (established 3297) is a single. The
parent's threshold moves 7 times inside the sub's life — to 0.57597 at 2659 (the LOH of H1 664): the very wick that
is sub 6's cycle-3 CTS, so that CTS is no longer strictly beyond → cycle 4 is cross-eligible (final `P` 0.5763,
3275). **The fix at establishment changed nothing:** on 0 of 28 cycles does the PERMISSION read with `P` at the
cycle's establishment differ from the one read with `P` as of the sub's end (`P` itself moved on 5 subs). 2 moves
fall in a cycle's pre-established window (sub 6 cycle 4: 3263, 3275 — no answer change; it was allowed from 2659).
Why the fix rarely matters: a sub's M15 CTS past the parent's threshold lies inside an H1 candle whose own wick is
past it too, so the parent's threshold moves to that wick at the hour's end — a counter sub's CTS can stay beyond
the live `P` only inside that hour (the LOH lag) or while the parent's reversal is pending (no threshold update);
the fixed and the end answers can differ essentially only for a cycle established inside such a window.

**A recurring shape:** a counter sub's CTS wick is often the very wick that moved the parent's BOS threshold (MS moves
it to a wick past it without a close) — equal prices, not beyond (strict, PLAN_I Q2): the sub needs a NEW CTS past
it, as the write-up says ("the limit increases meaning we need a new 1st CTS to be past it").

## 7. The design

### 7.1 `CrossFibLimit` (`zones/cross_cycle_fib.py`) — landing 1

- **The source** `parent_bos_threshold` joins `CROSS_FIB_LIMIT_SOURCES` (CL-5): a counter sub — its first record's
  parent cycle's BOS threshold, read live. `_LIVE_SOURCES = {"parent_bos_threshold"}`: the only source that may move.
- **Two fields** (named by the review — `start_price` read like `prev_sub_starting_price` / `starting_idx`; a moment
  is a `*_at` stamp in the Naming Standard): `price_at_start: Optional[float] = None` — `P` as of the structure's
  start; `price_moves: Tuple[Tuple[int, float], ...] = ()` — each `(known_at, price)`: the candle from which the
  move is KNOWN, in the coordinates of the frame the limit is used on (like the break fields), and the new `P`.
- **`price` = the LAST move's price, else the start's** — the limit object's own final value (up to the parent
  cycle's end): the R6-2 tripwire and the sweep log read it; the EXPORT reads `P` as of the structure's end instead
  (§7.2, §3 "readings").
- **`price_at(moment)`:** None → `price`; else the price of the last move with `known_at ≤ moment` (inclusive — a
  move is known AT its candle), else `price_at_start` (`price` when it never moved). **`moves_until(moment)`:** the
  moves with `known_at ≤ moment` (None → all) — the export's count.
- **`__post_init__` adds:** moves ⟹ source in `_LIVE_SOURCES` and `price` not None; `price_at_start` set ⟺ moves;
  `known_at` strictly increasing; `price == price_moves[-1][1]`. (Each move strictly FURTHER is the producer's
  assert — it knows the direction.)
- **`localized(offset)`** — restructured (today it returns early when there is no break, line 93, and the live
  source never has one): shift the break fields when set, every `known_at` when there are moves, and return `self`
  when there is neither (`test_plan_k_landing_a.py:667` pins `plain.localized(100) is plain`).

### 7.2 `FibTracker` (`zones/fib_tracker.py`) — landing 1

- **`self._cross_permission: Dict[(sid, cycle), bool]`** — a cycle's cross permission, FIXED at its
  `CTS_ESTABLISHED` (CL-1). Plain bools: MS deep-copies the tracker's `__dict__` at every snapshot
  (`_snapshot_state`, lines 602 / 614–615), so a restore undoes a fix with everything else.
- **`_limit_cycle_at(sid, price)`** — `M` for a given `P` (the loop of today's `cross_fib_limit_cycle`);
  `cross_fib_limit_cycle(sid, at=None)` = `_limit_cycle_at(sid, lim.price_at(at))` — the export passes the
  structure's end (below); the anchors are the tracker's as they stand (R6-6 unchanged).
- **`_limit_reaches(sid, target, at)`** ⟺ `P(at)` set and (`M(P(at))` None or `target ≤ M`).
- **`_cross_allowed(sid, target)`:** no entry → `CrossFibLimitError` (unchanged); the fixed answer when
  `(sid, target)` has one; else `_limit_reaches(sid, target, self._evaluated_at)`. Its three engine callers all run
  inside `_evaluating` (CTS_ESTABLISHED line 713, CTS_UPDATED 954, CTS_THRESHOLD_UPDATED 1380), so `_evaluated_at`
  is the handled event's moment there. NOT `_moment()`: that raises outside a handler, and 10 existing direct calls
  in `test_plan_i_s3_cross_fib_limit_applied.py` (lines 140, 142, 146, 324, 377, 379) read a constant limit outside
  one — for a constant limit the moment is irrelevant (`price_at` ≡ `price`); an UNFIXED cycle of a MOVING limit
  read with no moment raises (`CrossFibLimitError`: a live limit needs the reading moment; a fixed cycle returns its
  answer first).
- **At the establishment** (line 815): `cycle_id == 0 or not self._fix_cross_permission(sid, cycle_id)` — computes
  `_cross_allowed` and stores it; a cycle fixed twice raises. Why it cannot fire (the review traced it): MS's one
  establishment emission (line 1368, in `_apply_pattern_at_apply_idx`, after the void return 1344–1351) — the void
  snapshot is restored only on VOID, before any emission (898–903); the range look-aheads (938–945, 1550–1557) run
  `_replay_step_no_patterns`, which never establishes; `cts_cycle_id` only increments (1364); every FibTracker sees
  each (sid, cycle) once (the embedded one per MS run, `structure_engine` 84; the drawn one per
  `_run_downstream_pipeline`, orchestrator 312). Measured: 0 double establishments over the full suite (897 fixes).
  CTS_UPDATED (line 1015) then reads the fixed answer; the pre-established handler (line 1407) reads LIVE (its
  target is not established yet — never fixed), so a move can open a pre-established cross late.
- **`_structure_reference_limit(sid, target, at)`:** no entry / `P` None → None (unchanged); the fixed answer when
  there is one, else `_limit_reaches(sid, target, at)` (cycle 0 always reaches when `P` is set) — the answer the
  cycle's cross decision uses: CL-1 carried through PLAN_K's "the structure side applies exactly where a cross is
  allowed" (a derived reading, §3), so the IC reference's T ≤ M gate is fixed at establishment too.
- **`ic_reference`** passes `evaluated_at` to it; the watch reads `find_cross_limit_break(df,
  float(lim.price_at(evaluated_at)), sd, start, known_at=evaluated_at)` (CL-7: a break of a `P` the parent later
  moved past stops counting; `evaluated_at` None → the object's last `P`). The `prev_structure_bos` branch is
  unchanged (a reversal limit never moves). MS's in-flight POI inner refreshes only at a CTS_ESTABLISHED /
  CTS_UPDATED (`_refresh_poi_inners_for_cycle`, MS 1921–1941): a `P` move reaches MS's read at its next refresh —
  as a watch break does today.
- **The export, as of the STRUCTURE'S END** (§3 "readings" — CL-6): `_cross_fib_limit_meta(sid, at)` with `at` =
  the lifecycle cap, passed by BOTH `_finalize_lifecycle_fields` (line 691; `lifecycle_cap` in scope, 582) and
  `stamp_ic_references` (line 443 builds the tail-key list from it — the same `at`: a finalize read at None with the
  stamp at the cap would leave the conditional keys in the head; the reverse is harmless, since the stamp only
  re-orders and its key set at None is a superset — landing 1's mutant M12b, equivalent) — the IC reference's
  convention (`stamp_ic_references(lifecycle_cap)`). `cross_fib_limit_price`
  = `price_at(at)`, `cross_fib_limit_cycle` = `M` at that `P`, and — only when `moves_until(at)` is non-empty —
  `cross_fib_limit_price_at_start` + `cross_fib_limit_move_count` (= that count), right after
  `cross_fib_limit_from_sub_id`. The cap is in the frame's coordinates (the projection's `cap_local`), as the
  localized moves are; `at` None (the H1 main, an open sub) → the object's last `P` / every move.
- **Why L1 is byte-identical on every existing limit:** no move → `price_at` ≡ `price`, no new key; and the fixed
  answer equals today's re-computed one — the cycles k < T are confirmed before T exists (the order tripwire at line
  761; the confirmed anchor wins, line 998; one CTS_CONFIRMED per cycle; CTS_RECONFIRMED is not fed to the tracker),
  so today's `_cross_allowed(T)` is already constant over T's life. Measured over the full suite (review B): 0 of
  1790 `_cross_allowed` and 21,193 `_structure_reference_limit` calls differ between the fixed and today's answer.
- **A consequence, stated (CL-6):** a cycle fixed at its establishment can disagree with the exported `M` (as of the
  end) — a cycle established single (`via: "cross_fib_limit"`) before a move lifts `M` past it. Essentially only
  inside the windows of §6 ("Why the fix rarely matters"); census: 0 of 28 cycles.

### 7.3 The producer (`multitf/parent_tables.py`, `multitf/cross_fib_limit.py`, the orchestrator) — landing 2

- **`ParentTables.bos_threshold_steps[(S, C)]`** = `((h1_known_at, m15_known_at, price, event_type), …)` in
  processing order — every `BOS_CONFIRMED` / `BOS_THRESHOLD_UPDATED` of (S, C), the M15 moment by the TIMING mapper
  (`loh_fn`, LOH). Built once in `build_parent_tables`; a defaulted field (`test_lifecycle_sweep_unit.py:91`'s
  direct constructor keeps working).
- **`parent_bos_limit(tables, S, C, direction, s)`:**
  - asserts (raise, do not degrade — PART4 §17.7; each held on the 6 census windows): exactly one `BOS_CONFIRMED`
    (S, C), and it is the first step; its LOH ≤ `s` (the record's floor = LOH of max(the structure's start, the BOS
    moment)); no step at or after `end_h1[(S, C)]`; each threshold strictly further than the one before
    (`beyond(new, previous, direction)`);
  - steps at one M15 moment collapse — the last in processing order wins;
  - `price_at_start` = the last step's price with LOH ≤ `s` (CL-1: every read before the start uses `P` as of the
    start) — NOT the `BOS_CONFIRMED` price when earlier moves fold in (census 2025-04-01 → 05-20, subs 3 / 4: BOS
    0.59252, 9 earlier moves → 0.58938); `price_moves` = the steps with LOH > `s`; `price` = the last move's price,
    else the start's; `ref_anchor_price` = the `BOS_CONFIRMED` price (CL-3); no `from_pool_idx` (an H1 parent).
  - **The two R6-2 (a) shapes** (the fire's creating record ≠ the started_by record; the tripwire's `price_at_start`
    / `price_moves` comparison WARNs, the stored `P` stays — as for every source): (i) a zero-length creating record
    on sid S, the sub then started by a record under S+1 → the counter test can flip (the stored `P` may be a
    counter `P` on a sub exported as confluence, or the reverse); (ii) a later-fired record with an EARLIER start
    (a `first_confluence` / `reversal` finalize after its trigger) → `sub.start_idx` < `s`, so `price_at_start`
    folds the moves of (`sub.start_idx`, `s`] — reads in that span see them early. The post-landing census reports
    the tripwire's mismatch count per window (§9; expected 0).
- **`cross_fib_limit_for(…, parent_tables)`** (a required keyword): FIRST, `if int(direction) !=
  int(parent_tables.parent_sd[S])` → `parent_bos_limit(…)` — the sweep's own `relative_dir` formula (line 371), for
  every trigger type (CL-0 "regardless of the sub trigger"); else the existing branches, unchanged.
- **`check_cross_fib_limits(pool, h1_events, parent_tables, *, log)`** hands the tables on and compares
  `price_at_start` + `price_moves` too.
- **The orchestrator:** `_resolve_cross_fib_limit` passes `parent_tables=tables`; `check_cross_fib_limits(sub_pool,
  sorted_events, tables)`. The sweep's `[sweep] cross_fib_limit` log line adds `moves=N` when it moved.
- **Look-ahead, stated:** the moves are read off the WHOLE H1 stream at the fire (the sub's geometry is built whole
  at its start, as today); each is USED only from its LOH moment on (`price_at`) — the pattern of the reversal
  limit's realised break (PLAN_K RX1-K7) and the parent tables' ends.

### 7.4 The hover (`charting/_zone_render.fib_hover_limit_text`) — landing 1

When the record carries `cross_fib_limit_move_count`, the move qualifies the PRICE: `limit 0.57630 (moved 7× from
0.57328) · parent_bos_threshold · M=4` (census 2025-02-01 → 03-22, sub 6 as of its end 3307: cycle 3's CTS 0.57597 is
not beyond 0.5763, cycle 4's is — the review corrected the draft's `M=none`). Unchanged otherwise. (No supplier
label: `parent_cts_threshold` shows none today either.)

### 7.5 Unchanged

The H1 main's limits (`compute_main_cross_fib_limits`), the probe's `CrossFibLimit(None, "none")`, `reversal_limit` /
`parent_cts_limit` / `read_prev_sub` (they now serve the CONFLUENCE subs only), MarketStructure (its embedded tracker
runs the same FibTracker code on the same localized limit — `entity_df_mutation` lines 1167 / 1300 already hand
`lim.localized(slice_begin)`), `_FIB_META_IDX_KEYS` (the two new keys hold no candle index) — but
`cross_fib_limit_move_count` is an int, so it joins `NON_INDEX_INT_META_KEYS` (`tests/test_event_meta_idx_keys.py`
371–379, next to `cross_fib_limit_cycle`; the guard at 687 fails any int meta key in neither list — and the
reference fixture never moves, so only the registration keeps the guard honest).

### 7.6 The audit sites (every reader of the limit, grep 2026-10-11)

| Site | Reads | Change |
|---|---|---|
| `fib_tracker.cross_fib_limit_cycle` | `lim.price`, the anchors | none — the FINAL `M` (export) |
| `fib_tracker._cross_allowed` (callers 815 / 1015 / 1407) | `P`, `M` | live until fixed (L1) |
| `fib_tracker._structure_reference_limit` | `P`, `M` | the same answer (L1) |
| `fib_tracker.ic_reference` (the watch) | `lim.price` | `price_at(evaluated_at)` (L1) |
| `fib_tracker._cross_fib_limit_meta` ← `_finalize_lifecycle_fields` 691, `stamp_ic_references` 443 | the keys | as of the lifecycle cap (the same `at` at both); + 2 keys when moved (L1) |
| `tests/test_event_meta_idx_keys.py` 371–379 (`NON_INDEX_INT_META_KEYS`) | int meta keys | + `cross_fib_limit_move_count` (L1) |
| docstrings that say `P` never moves: `cross_limit_watch.py` 12–15, `CrossFibLimit` (cross_cycle_fib 43), `fib_tracker` 255–257 + `_cross_allowed` 329–330, `cross_fib_limit.py` 1–2 | — | reworded (L1 / L2) |
| `_zone_render.fib_hover_limit_text` | the meta | + the suffix (L1) |
| `entity_df_mutation` 1167 (geometry) / 1300 (projection) | `localized` | moves shifted (L1) |
| `lifecycle_sweep` 320–335 (fire), 544–548 (`replace(from_sub_id)`) | the limit | log line (L2); `replace` keeps the new fields |
| `cross_fib_limit.cross_fib_limit_for` / `check_cross_fib_limits` | — | the counter branch, the tables (L2) |
| `orchestrator` 1005–1012 / 1026 | — | pass the tables (L2) |
| `structure_engine` 52–86 / 205 / 344–405 | pass-through | none |
| `structure_lifecycle.compute_main_cross_fib_limit(s)` | the H1 main | none |
| `unified_probe` 553 | no limit | none |
| `plan_m_inputs/limit_table.py` (tool) | the meta | + the 2 keys |

## 8. Stages and predictions (one cause per `/compare`)

**Landing 1 — the live-`P` plumbing, no producer** (§7.1, §7.2, §7.4). Nothing yet makes a limit move →
**prediction: `/compare` vs S_L2 byte-identical** — every CSV's md5, and the figures including the hovers
(`cmp_save.py` + `cmp_hover.py`, + `plan_m_inputs/fig_hash.py` equal to S_L2's). **And the census on L1:** the six
windows of §6 re-run on the landed L1 code — every run completes (L1 adds a raise to every structure: H1, both
lenses' subs, the geometry MS, the probe's MS) and every `[census]` line equals §6's run (`census_pre_build.txt`'s
inputs). Tests: §9 L1. Commit on the user's go; save S_M1.

**Landing 2 — the counter-sub source** (§7.3). **Prediction: IDENTICAL to the trial's own outputs** (§4; the review
showed the trial = the design on this window: the same `P`, source, anchor, no `from_*`, no new key and no log change
without a move; the tables' `parent_sd` = the trial's CTS_ESTABLISHED directions):
- every CSV's md5 = `plan_m_inputs/trial_md5.txt` (27 CSVs; vs S_L2 exactly the 6 M15 ones differ — both lenses' fib
  lifecycle, IC candidates, POIs; copies in `plan_m_inputs/trial_csvs/` for a diff if it fails);
- each figure's `fig_hash.py` line = `plan_m_inputs/trial_fig_hash.txt` (the full figure, hovers included; the H1
  chart's = S_L2's);
- `trial_diff.py` S_M1 vs the L2 replay = `plan_m_inputs/trial_diff_vs_S_L2.txt` (the human-readable delta,
  regenerated whole after the review found the first copy truncated): sub 3 cycle 2 becomes a cross from cycle 1
  (pre-established 2967), its POI's IC 2969 → 2821, the `via` relabels, the IC-reference moves;
- run.log: the `[sweep] cross_fib_limit` lines of subs 1 / 3 / 5 / 7 at 0.55808 / 0.5736 / 0.58105 / 0.58105, no
  `moves=`; the R6-2 tripwire "3 subs re-derived, 0 mismatches".
**And the census on L2 — the PINNED table** (`census_report.py`'s, the design's definitions; window = the config
start): `P` at start / the object's moves → its last `P` / the export's moves → `P` as of the end —
2025-02-01: sub 4 0.56777 / 2 → 0.56731 / 2 → 0.56731; sub 6 0.57328 / 7 → 0.5763 / 7 → 0.5763;
2025-04-01: subs 3 and 4 0.58938 / 9 → 0.58466 / 9 → 0.58466;
2025-08-01: sub 3 0.59149 / 13 → 0.58538 / **0 → 0.59149** (all 13 after its end); sub 5 0.59149 / 13 → 0.58538 /
13 → 0.58538; every other counter sub: its BOS price, no move (reference 0.55808 / 0.5736 / 0.58105 / 0.58105;
2025-02-01 sub 1 0.55164; 2025-06-01 sub 1 0.58831; 2025-10-01 subs 1, 3 0.5845) — the `[sweep]` lines show the
object, the fib meta the export; the tripwire 0 mismatches on every window; 2025-02-01 sub 6 cycle 4 cross-eligible
(a fib record of it, if any, is not labelled `via: "cross_fib_limit"`). The pin holds unless the knock-on of §1 item 1
changes a counter sub's own structure (MS reads `P`) — then the row is explained, not forced. Structure-event and
confluence-sub changes vs a HEAD run listed separately (no prediction, reported). Chart review; commit on the user's
go; save S_M2.

## 9. Tests, mutants, the census

**Landing 1 (synthetic; `tests/_event_factory.py`, `tests/_cross_fib_limits.py`; every test asserts ABSOLUTE
outcomes, not only parity):**
1. `CrossFibLimit`: `price_at` — None, before the first move, AT a move's `known_at` (inclusive), between, after;
   `moves_until`; `localized` on a NO-break limit with moves (the real shape — the live source never has a break)
   moves every `known_at` and keeps the prices, a limit with neither is returned as `self`; each new assert raises.
2. `FibTracker`, a structure whose `P` moves further mid-run:
   a. pre-established: a `CTS_THRESHOLD_UPDATED` of cycle T−1 before the move → no pre-established cross; AFTER it
      → the cross starts (late) — the test reads both sides (it kills "cache the first read");
   b. fixed at the establishment: T established single (`via: "cross_fib_limit"`) before a move stays single on a
      later `CTS_UPDATED`; T established as a cross keeps it;
   c. the snapshot: deep-copy the tracker's `__dict__`, establish T, restore, move, re-establish → the NEW answer
      (the restore removed the fix); a second establishment without a restore raises;
   d. `_structure_reference_limit`: an established T → its fixed answer at every `evaluated_at`; a pre-established T
      → live at its `evaluated_at`; a T that never establishes, read with `at=None` → the object's last `P`; cycle 0
      → always;
   e. the watch: a break of the earlier `P` counts at a moment before the move and stops counting after it; a break
      of the last `P` counts on both sides once confirmed;
   f. the export as of the cap: a cap BEFORE a move → no move keys, `cross_fib_limit_price` = `P` at the cap, `M` at
      that `P`; a cap after → both keys (`move_count` = the moves ≤ cap); `at` None → every move; the EXACT
      `list(meta)` order after `_finalize_lifecycle_fields` + `stamp_ic_references` (the limit keys last, the two new
      keys right after `cross_fib_limit_from_sub_id`);
   g. a moving limit read with no moment raises; a constant one does not (the 10 existing outside-handler calls);
   h. the hover: `limit 0.57630 (moved 7× from 0.57328) · parent_bos_threshold · M=4`.
3. The REAL collaborator: a real `MarketStructure` run, `test_plan_i_s5_ms_reads_the_fib._rows(sd)` (cycles 0–3,
   one 0→1 cross, 13 handled events — measured at build; the draft said "cycles 0–1"), `P` at the start just inside
   cycle 0's CTS (so `M` = 0), one move to exactly that CTS price (not strictly beyond), its `known_at` taken from the
   run's own events: a move AT cycle 1's establishment moment (10, inclusive) → the cross; one candle AFTER → the
   single, `via: "cross_fib_limit"`, its IC reference CTS_0 only, exported `M` = 1 (cycle 1's CTS 0.6167 is beyond
   the moved `P` 0.6122 — the draft's "`M` none" was wrong; it is §7.2's stated consequence) — asserted in BOTH
   trackers (MS's embedded read via `representative_fib` + `ic_reference`, and the drawn one); plus the
   `start_abs=60` geometry of `test_plan_k_landing_a.py:705` → the moves arrive shifted by `slice_begin` = 10
   (`entity_df_mutation` 1167; the draft said 50).
4. **Intended edits:** `test_plan_i_s2_cross_fib_limit.py:78` (the exact `CROSS_FIB_LIMIT_SOURCES` set +
   `parent_bos_threshold`); `test_event_meta_idx_keys.py` 371–379 (+ `cross_fib_limit_move_count`). Nothing else
   may break (review B: L1's fixed answer = today's on every fixture).

**Landing 2:**
1. `parent_bos_limit`: `price_at_start` folds the moves with LOH ≤ `s` (inclusive) — a fixture where it differs from
   the BOS price — keeps the later ones, collapses one M15 moment (the last wins; `move_count` after the collapse),
   the last `price`, the anchor = the BOS price; each assert (two BOS, a BOS after `s`, not further, a step at the
   cycle's end); steps filtered by BOTH sid and cycle (a fixture with ≥ 2 sids × ≥ 2 cycles at distinct prices).
2. `cross_fib_limit_for`: counter → `parent_bos_threshold` for `reversal` / `first_counter` / `subsequent_counter`; a
   confluence reversal still → `reversal_bos_frozen`; `first_confluence` unchanged; `parent_sd` read for the RIGHT
   sid (a 2-sid fixture with opposite directions).
3. `ParentTables.bos_threshold_steps`: the injected `loh`, processing order.
4. The R6-2 tripwire: a `price_at_start` / `price_moves` mismatch is counted.
5. The reference window on the REAL H1 events, driven by the committed S_L2 `*_M15_*_subs.csv` (each sub's first
   record's parent, direction, `start_idx`) for ALL 8 subs — not a hand-coded list: subs 1 / 3 / 5 / 7 counter
   (`P`, source, anchor, no moves), and the negatives — sub 2 (a confluence reversal), subs 0, 4, 6 — keep their
   sources.
6. **Intended edits** (review B): `test_plan_i_s2_cross_fib_limit.py` 427 (lines 430–440: the `parent_tables`
   keyword; 433 / 435–436 flip meaning — direction +1 under sid 1's −1 is now counter), 764 (767–771), 527 / 822 /
   857 (`check_cross_fib_limits`' tables argument), 916 (the `build_parent_tables` stub returns None);
   `test_plan_k_landing_a_review_pins.py:132`. TRAP: the shared `_rec` fixture hard-codes parent (0, 0) and
   `relative_dir="confluence"` for both directions — the sell reversal subs Y / y need `parent_sd[0] = -1`, b `+1`,
   or they silently take the counter branch and the reversal re-derivation tests test nothing.

**Mutants** (`mutant_runner.py` house style — `plan_h_inputs/mutants_*.py`; the full suite; each must be killed,
CALL SITES included). L1: `_cross_allowed` reads `lim.price`; caches its first read; the fix never stored; a second
fix silently keeps the first; the fixed answer ignored by `_structure_reference_limit`; the watch reads `lim.price`;
`price_at` `<` for `≤`; `localized` leaves the moves / always replaces; the export reads the last `P` instead of `P`
at the cap; only one of the two export calls passes the cap; the keys always / never / misplaced; the hover suffix
dropped. L2: the counter test inverted / skipped for `reversal`; `parent_sd` of the wrong sid; `price_at_start` with
`<` for `≤` / = the BOS price; moves not collapsed / counted before the collapse; steps filtered by sid only / by
cycle only; the anchor = `P`; the tripwire ignores the moves; the price-extreme mapper for LOH.

**The census** (§8): on L1 (every `[census]` line = §6's run) and on L2 (the pinned table). The census patch's anchor
line changes with L2's `check_cross_fib_limits` signature — re-pointed then. **L2 adds one count** (§3 "readings",
CL-7's other side): per counter sub with a move, the IC watch's break of each moved `P` — how many confirmed BEFORE
the move that made them count (expected rare: only inside the LOH lag or while a reversal is pending).

## 10. Docs at each landing

- **L1:** CROSS_CYCLE_FIB_SPEC §3 (a live source: `price_at`, the permission fixed at establishment, the export as
  of the structure's end, the watch at `P(t)`) + §8 (MS reads the same, at its next POI refresh); POI_ZONES_SPEC §3.2
  (the watch's threshold as of the reading moment); CHARTING_SPEC (the hover's `(moved N× from …)`); GLOSSARY
  (`cross_fib_limit_price_at_start`; `cross_fib_limit_move_count` — the moves AFTER the sub's start and up to its
  end; the start may already fold earlier ones); the code docstrings that say `P` never moves (§7.6).
- **L2:** CROSS_CYCLE_FIB_SPEC §3's source table + the counter rule; PLAN_I §1.2's pointer → built; GLOSSARY (counter
  sub, `parent_bos_threshold`); the register §D.1 item 2; MEMORY.md priorities + the session log; the `/compare`
  skill's save-format boundaries (S_M1, S_M2).

## 11. The cold review (2026-10-11) — DONE, every finding folded

Three fresh read-only reviewers (no worktree; estimated 500–650K, actual **760,821 tok**: A 262,765, B 279,422,
C 218,634): (A) the design against the CODE; (B) the tests, mutants and predictions — what could pass while wrong;
(C) the design against the user's write-up and CL-0…CL-8.

| Finding (lens, severity) | Folded into |
|---|---|
| The export would carry moves made AFTER the sub ended (2025-08-01 sub 3: 13 moves, all after its end) — A, B, C: MAJOR | §3 "readings" (CL-6's "final" = as of the structure's end), §7.2 the export, §8 pinned table |
| L2's "trial_diff line for line" can pass while wrong; the stored diff was truncated — B: BLOCKER (on the check) | §8: md5 = the trial's 27 CSVs (`trial_md5.txt`), full-figure hashes (`trial_fig_hash.txt`, `fig_hash.py`), the 6 changed CSVs persisted (`trial_csvs/`), the diff regenerated whole |
| `_moment()` raises outside a handler (10 existing test calls); the source-set pin; `localized(...) is plain` — A, B: MAJOR / MINOR | §7.2 (`_evaluated_at`; a moving limit with no moment raises), §7.1 (`localized`), §9 L1 item 4 |
| `localized` returns early with no break — the live source never has one — A: MINOR | §7.1 |
| `cross_fib_limit_move_count` must join `NON_INDEX_INT_META_KEYS` — A: MINOR | §7.5, §7.6, §9 |
| Creating vs started_by record (two R6-2 (a) shapes) — A, C: MINOR | §1 item 1, §7.3 |
| `rv_anchor_failed` moves carry MS's look-ahead moment — A: MINOR | §1 item 2 |
| The real-collaborator pin could pass on shared bugs and never localizes — B: MAJOR | §9 L1 item 3 (absolute outcomes, `start_abs=60`) |
| Mutants / tests missing (first-read cache, silent second fix, folded start, sid/cycle filters, wrong `parent_sd`, count before collapse, key order) — B: MAJOR | §9 |
| The L2 test edits + the `_rec` fixture trap; test 5 circular → drive from the S_L2 subs CSV — B: MINOR | §9 L2 items 5–6 |
| Census wording (cycles before the start; "permission", not `P`) + the report's model — B, C | `census_report.py` fixed and re-run, §6 |
| The hover example's `M=none` → `M=4`; "(moved …)" qualifies the price; the field names — C: MINOR | §7.1, §7.4 |
| The break detector chosen silently (the M15 watch vs sid 1's H1 reversal pattern) — C: MINOR | §1 item 4, §3 "readings" |
| The IC gate's fix credited to CL-3 — C: MINOR | §3 "readings", §7.2 |
| Knock-on: a counter sub's MS reads `P` → other subs' values can change — C: MINOR | §1 item 1, §8 (listed separately) |
| Run the census on L1 too (L1 adds a raise everywhere) — C: MINOR | §8 |
| "A cycle fixed twice" — the stated reason was vacuous — A: NOTE | §7.2 (the traced facts + 0 measured) |
| MS sees a `P` move at its next POI refresh; docstrings still say `P` never moves — A: NOTE | §7.2, §7.6, §10 |
| §2's exception timing (the LOH lag), §1's "Why", CL-2 covers the next sid, label clarity — C: NOTE | §1, §2 |

Verified by the reviewers as correct: the audit sites otherwise complete; L1's byte-identity argument (0 differing
answers over the full suite); the snapshot copy; LOH(BOS) ≤ `s`; no threshold step at / after a cycle's end (MS's
barrier order); the coordinates of every `evaluated_at` vs the localized moves; MS-parity; every §2 / §4 / §6
number against the input files; L2 = the trial on the reference window.

## 12. Landing 1 — as built (2026-10-11, session 1; COMMITTED `86d4f7b`, saved S_M1 `20261011_204047_86d4f7b`)

The user (2026-10-11): **"a, go ahead with landing 1"** (§3 "readings": the export's "final" = as of the structure's
end). Built on `27ab6da` (= S_L2's engine + docs).

**Code** (§7.1 / §7.2 / §7.4, as designed): `zones/cross_cycle_fib.py` (`parent_bos_threshold` in the source set,
`_LIVE_SOURCES`, `price_at_start` / `price_moves`, `price_at` / `moves_until`, the asserts, `localized` restructured);
`zones/fib_tracker.py` (`_cross_permission`, `_limit_cycle_at`, `cross_fib_limit_cycle(sid, at)`, `_limit_reaches`,
`_cross_allowed` on `_evaluated_at` with the fixed answer first and the no-moment raise, `_fix_cross_permission` at
the establishment, `_structure_reference_limit(sid, target, at)`, the watch on `price_at(evaluated_at)`,
`_cross_fib_limit_meta(sid, at)` + the two keys, `at = lifecycle_cap` at both export calls);
`charting/_zone_render.fib_hover_limit_text` (the suffix); docstrings (`cross_limit_watch.py`, the tracker's class and
`_limits` comment). Tool: `plan_m_inputs/limit_table.py` prints `(moved Nx from …)` when present (its S_L2 output
unchanged); `fig_hash.py` takes an optional NAME_PREFIX (the repo's `artifacts/charts` holds many runs).

**`/compare` vs S_L2 — prediction (§8) MET, byte-identical:** fetch gate PASS (4228 M15 candles, 5 chunks); replay
48.6 s wall (multi_tf_dual 8.70 s, charts 6.44 / 7.93 / 20.37 s); `cmp_fig_full.py`: 27 / 27 CSVs md5-equal, the three
figures 0 differing traces (every attribute), layouts equal; `cmp_hover.py`: 0 hovertemplate / customdata changes;
`fig_hash.py` = S_L2's on all three charts (`plan_m_inputs/l1_fig_hash.txt`; the H1 line = the trial's, as predicted).

**The census on L1 — MET:** the six windows of §6 re-run on HEAD + the L1 engine files (`census_l1_run.txt`): every
run exit 0 (L1's raises never fire), every window's `[census]` lines md5-equal to the pre-build run.

**Tests** (`tests/test_plan_m_landing_1.py`, 45, §9 L1 items 1–3; item 4's two intended edits): suite **1654 passed**
(1609 + 45; the OANDA smoke included) — after §13's fix-up 1657 (48). One pin added during the mutant listing: `ic_reference` hands its reading
moment to `_structure_reference_limit` for an unfixed (pre-established) target — without it M14 survives.

**Mutants** (`plan_m_inputs/mutants_m1.py`, the full suite each, `mutants_m1_result.txt`): **25 / 26 KILLED**; the
survivor M12b (the stamp's tail list read at None) is EQUIVALENT, declared before the run (§7.2: the stamp only
re-orders, and its key set at None is a superset of the one at the cap).

**Plan text corrected by measurement:** §9 L1 item 3 (the fixture has cycles 0–3; the exported `M` in the
"move after" case is 1, not none — §7.2's stated consequence; the geometry shift is `slice_begin` = 10, not 50);
§7.2 (which of the two export calls must pass the cap — finalize; the stamp's is equivalent).

**Docs (§10 L1):** CROSS_CYCLE_FIB_SPEC (status line, §3 "A moving `P`", the export, the watch, the structure
reference, §8 MS's read); POI_ZONES_SPEC §3.2 (the watch at `P` as of the reading moment); CHARTING_SPEC (the hover's
moved suffix); GLOSSARY (a moving `P`, the two keys, `parent_bos_threshold` in the source list, `M` as of the end).

## 13. Landing 1's review (2026-10-11) — two fresh read-only reviewers, no worktree (estimated 350–500K tok)

(A) the code against the design, byte-identity on a constant limit, the raises' reachability, the docs; (B) the
tests and mutants — what could pass while wrong.

**Lens B (190,214 tok): 0 BLOCKER, 0 MAJOR.** Suite 1654 on the working tree; only the two intended pre-existing test
files changed. Its 12 extra mutants (`%TEMP%/rvB_m1/mutants_extra.py`): 8 killed; X01 / X02 / X04 SURVIVED; M12b
re-run SURVIVED — equivalent, agreed (F3).

| Finding (severity) | Folded into |
|---|---|
| F1 (MINOR): the PROJECTION call site's `localized` (`entity_df_mutation` 1300) unpinned with a moving limit — X04 (localize only a limit with a break) survived; in L2 the drawn tracker would read every move `slice_begin` candles late (its permission could differ from MS's; the export would undercount) | new pin `test_the_projection_hands_the_drawn_tracker_the_slice_local_moves` (the drawn tracker's stored moves shifted by 5; the mirrored records' keys counting a move known 2 candles before the cap) |
| F2 (MINOR): "a cycle established as a cross keeps it" could not fail — with further-only moves a fixed True never disagrees with the live answer; X01 (a True fix re-read live) / X02 (the structure reference honours only a False fix) survived | `test_a_cycle_established_as_a_cross_keeps_it` now uses a NARROWING move (1.30 → 1.35 at 58 → 1.20 at 63: live at 64 refused, `M` = 1) — unreachable in production (L2 asserts further-only) but accepted by `CrossFibLimit`; asserts the live refusal, the kept cross, and the structure side kept |
| F4 (NOTE): `NON_INDEX_INT_META_KEYS`'s new entry unexercised until a real moving limit (L2) | known (§7.5) — L2's census / trial |
| F5 (NOTE): a constant `parent_bos_threshold` export not pinned directly | L2's trial md5 (subs 1 / 3 / 5 / 7 are constant counter limits) |
| F6 (NOTE): CTS_UPDATED on an unfixed cycle ≥ 1, and the MS tracker's finalize — unreachable / absent | none |

The three survivors re-run against the new pins: `plan_m_inputs/mutants_m1_review.py` — **3 / 3 KILLED**
(`mutants_m1_review_result.txt`; X01 / X02 by the narrowing-move pin, X04 by the projection pin). Tests now 46 in
`test_plan_m_landing_1.py`.

**Lens A (260,854 tok): 0 BLOCKER, 0 MAJOR** — the code = §7.1 / §7.2 / §7.4; measured, with every answer compared to
HEAD's re-computed one: the full suite (`_cross_allowed` 1,790 calls / `_structure_reference_limit` 21,193 / the export
812 — 0 differ; 898 fixes, 0 twice) and the reference replay (27 / 27 md5 = S_L2; 333 / 538 / 54 calls, 0 differ; 36
fixes); the raises traced unreachable in production; a stress replay with synthetic moves on EVERY priced limit exits
0 (82 live reads, the move keys on 40 / 54 exports, sub 4's post-end move correctly left out).

| Finding (severity) | Folded into |
|---|---|
| MINOR-1: CROSS_CYCLE_FIB_SPEC's watch bullet "the bound never moves" unqualified | qualified: for one `P`; a moving `P` watched at `price_at(t)` (both sides of CL-7) |
| MINOR-2: GLOSSARY "cross-limit breakout watch" — "`P` fixed" | "one `P` per watch … a MOVING `P` at `P` as of the reading moment" |
| NOTE-3: "fixed once" unqualified (CROSS_CYCLE_FIB_SPEC §0 / §3 heading); the source table, the `ref_anchor_price` table, the `CrossFibLimit` docstring, `cross_fib_limit.py`'s header lack the new source | §0 + §3 heading qualified now; the rest is L2's (§7.6 / §10 — the producer's rows) |
| NOTE-4: POI_ZONES_SPEC §3.2's structure-reference gate silent on the fix | added: fixed at T's establishment, else live |
| NOTE-5: `entity_df_mutation` 1166 / 1298 comments say `localized` shifts the break only | + "each move's `known_at`" |
| NOTE-6: "a moving limit read with no moment raises" broader than the code (a fixed cycle answers first) | reworded (docstring, spec, §7.2): an UNFIXED cycle |
| NOTE-7 (design): from a move's moment the earliest break of the NEW `P` counts, even one confirmed before the move | §3 "readings" (derived from CL-7 + K6, shown to the user); the L2 census counts it (§9); the spec's watch bullet states it |
| NOTE-8 (design): nothing in L1 enforces "further only" — a nearer move could leave a pre-established cross active beside a fixed single | `_fix_cross_permission` RAISES on a refusal over an active pre-established cross; pin `test_a_refusal_over_an_active_pre_established_cross_raises`; mutant M25 |
| NOTE-9 (API): `at=None` defaults silently read the last `P` (a look-ahead) | `_structure_reference_limit` and `_cross_fib_limit_meta` take `at` REQUIRED; the public `cross_fib_limit_cycle(sid, at=None)` keeps its default (the final `M` after the stream; ~10 test calls) |

**After the fix-up (the final L1 code):** suite **1657 passed** (`test_plan_m_landing_1.py` 48); replay re-run —
fetch gate PASS, `cmp_fig_full.py` 27 / 27 md5-equal + the three figures identical, `cmp_hover.py` 0, `fig_hash.py`
= S_L2's; the census re-run on the six windows — every `[census]` line equal, no raise (`census_l1_run.txt`); the
review mutants `mutants_m1_review.py` **4 / 4 KILLED** (X01, X02, X04, M25). Review tokens: **451,068** (A 260,854,
B 190,214; estimated 350–500K).

**Committed** on the user's go (2026-10-11, "sounds good go ahead"): source `86d4f7b`; `/commit-save` run mode —
replay 47.5 s wall (multi_tf_dual 9.02 s; charts 6.38 / 7.62 / 20.72 s), FETCH GATE PASS; the save S_M1
`20261011_204047_86d4f7b` (27 CSVs + 3 charts; `cmp_fig_full.py` vs S_L2: 27 / 27 md5-equal, the figures identical);
save commit `8bdcc8a`, trunk `a442e91`; `LATEST_<branch>` = S_M1. Local, NOT pushed.

## 14. Landing 2 — as built (2026-10-11, session 1; COMMITTED `10e4c6f`, saved S_M2 — §16)

Built on `b5a3230` (= S_M1's engine + the pointers) after the user's go on landing 1 ("sounds good go ahead").

**Code** (§7.3, as designed): `multitf/parent_tables.py` — `ParentTables.bos_threshold_steps[(S, C)]` =
`((h1_known_at, m15_known_at, price, event_type), …)`, processing order, `ef.event_moment` + the injected LOH, a
defaulted field; an unmapped step (its hour before the M15 frame) or an unpriced one (a synthetic fixture:
`test_parent_tables`' BOS events carry no price — found by the suite, not in the plan) is STORED, and asserted only
where it is read. `multitf/cross_fib_limit.py` — `parent_bos_limit(tables, S, C, direction, s)` (the asserts, the
fold into `price_at_start`, the collapse, the anchor = the BOS price); `cross_fib_limit_for(…, parent_tables)` checks
the counter test FIRST; `check_cross_fib_limits(pool, h1_events, parent_tables, *, log)` compares `price_at_start` +
`price_moves` too (the set of re-derived subs unchanged — the trial's "3 subs"). The orchestrator passes the tables to
both; the sweep's `[sweep] cross_fib_limit` line ends `moves=N` on a moving limit.

**`/compare` — the prediction (§8) MET:** fetch gate PASS; all 27 CSVs' md5 = `trial_md5.txt`; the three figures'
`fig_hash.py` = `trial_fig_hash.txt`; `trial_diff.py` S_M1 vs the replay = `trial_diff_vs_S_L2.txt` (identical but
the CRLF of the stored copy); run.log — `[sweep] cross_fib_limit` pool#1 / #3 / #5 / #7 (= subs 1 / 3 / 5 / 7) P =
0.55808 / 0.5736 / 0.58105 / 0.58105 `parent_bos_threshold`, no `moves=`; "3 subs re-derived at their start, 0
mismatches". (That replay ran before the unpriced-step tweak — production BOS events are always priced; the final
replay re-runs at the commit.)

**The census on L2 — the PINNED table MET** (`census_l2_report.py` on `census_patch.py` + `census_run.py --export`,
the engine's OWN object and exported values; `census_l2_report.txt`, `census_l2_run.txt`): 14 / 14 counter subs MATCH
(`P` at start, the object's moves → its last `P`, the export's moves → its `P`); the tripwire 0 mismatches on every
window (3 / 2 / 2 / 0 / 3 / 4 subs re-derived); **0 knock-on** (no sub's cycles / life / non-counter `P` changed vs the
pre-build run; the H1 facts equal); **CL-7's other side: 0** (no exported break of a moved `P` confirmed before the
move); 2025-02-01 sub 6 cycle 4: `via: "cross_failed"` on both lenses — cross-eligible now, as pinned, falling back
for want of a live earlier cycle. **One pinned row's EXPORT is unseen:** 2025-08-01 sub 3 (the CL-6 case) has no fib
record — its cycle 0 is established at 3283, after the sub ended (3247 → 3259) — so its "export stays 0.59149" is
pinned by landing 1's unit tests only (§9 L1 2f); its object row matches.

**Tests:** `tests/test_plan_m_landing_2.py` (29): `parent_bos_limit` (the fold inclusive at the start, the
collapse, the anchor, the decoy cycles of two sids, each assert), the dispatch (counter first for every trigger
type, the right sid, a confluence sub's rules — a real reversing sub), `bos_threshold_steps` (processing order, the
injected LOH, an unmapped step), the sweep's `moves=` line, the tripwire's two new fields, and the reference window
on S_L2's REAL rows (16 H1 events + the 8 subs EMBEDDED — a mutant tree has no `artifacts/`; a provenance test
re-reads the save when present). The 7 intended edits (§9 L2 item 6), the `_rec` trap handled (`_tables` helper:
sid 0's direction keeps the fixtures' subs confluence). Suite **1686 passed**. While writing the real-data test I
typed sub 4's anchor from memory (0.57764 — sid 1 cycle 2's CTS_ESTABLISHED price); the extraction showed the rule
reads cycle 1's CTS_CONFIRMED, 0.574 — corrected before the first run.

**Mutants** (`plan_m_inputs/mutants_m2.py`, the full suite each, `mutants_m2_result.txt`): **25 / 25 KILLED** (the
counter test inverted / skipped for reversal / on sid 0 / after the reversal rule; the start exclusive / = the BOS;
no collapse / the first wins / moves from the raw steps; the anchor = `P`; the key without the cycle / the sid; each
assert dropped or loosened; the tripwire's two fields; the table's mapper / types / key; the driver's tables; the log).

**Docs (§10 L2):** CROSS_CYCLE_FIB_SPEC (status line; §3 the COUNTER row first in the source table, the formula
paragraph + asserts + reference values, the `ref_anchor_price` row); PART4 §17.6's tables listing
(`bos_threshold_steps`); GLOSSARY (**counter sub**, the source list, "a moving `P`"); the `CrossFibLimit` docstring's
anchor list (lens A NOTE-3); `cross_fib_limit.py`'s module header; PLAN_I §1.2's pointer → BUILT. At the commit: the
register, MEMORY.md, the session log, the `/compare` boundary (S_M2).

## 15. Landing 2's review (2026-10-11) — two fresh read-only reviewers, no worktree (estimated 350–500K tok)

**Tokens: 437,943** (A 246,123 — the producer against §7.3, the rule, the docs; B 191,820 — the tests and mutants).
**0 BLOCKER, 0 MAJOR.** Lens A measured: an H1 scan of 29 real windows (NZD_USD 17 two-month windows 2023-01 →
2025-11, EUR_USD / AUD_USD / GBP_USD 4 each): 99 cycles, 89 `BOS_THRESHOLD_UPDATED` (58 `probe_no_break`, 31
`rv_anchor_failed`), `parent_bos_limit` at the floor and at infinity on all 92 non-degenerate cycles — 184 calls, 0
asserts; its own reference replay on the final tree = the trial (27 / 27 md5, the figures, the `[sweep]` lines); the
census report re-run identical. Lens B: suite 1686; 9 extra mutants, 5 killed.

| Finding (lens, severity) | Folded into |
|---|---|
| A MINOR-1: "a reversal start = `prev_structure_bos`" stale (a COUNTER reversal sub has `cross_limit_anchor`) — `fib_tracker` docstring, `IC_REF_SOURCES` comments, GLOSSARY "IC reference", POI_ZONES_SPEC §3.2 | each reworded: `prev_structure_bos` = a `reversal_bos_frozen` limit (main / a confluence sub's reversal); `cross_limit_anchor` = every other source, the counter sub's included |
| A MINOR-2: GLOSSARY `cross_fib_limit_price` (no counter-first) and `ParentTables` (no `bos_threshold_steps`) | both rows |
| A MINOR-3: CROSS_CYCLE_FIB_SPEC's watch reference values stale after L2 | measured on the L2 replay's fib CSVs: subs 3 and 4 → 3588 (3589), subs 1 / 5 / 6 / 7 → none |
| A NOTE-4: the x−1 rule now reachable only by `subsequent_confluence` (the two counter types are counter by construction — `uc1_trigger` / `subsequent_counter_pipeline` set `lower_sd = −parent`; B measured 10 / 10 counter-typed subs counter on the census) | the spec's x−1 row, the module header, `PREV_SUB_TYPES`' comment, the tripwire docstring; the hover example (sub 7's old `prev_sub_bos_threshold … M=2` can no longer occur) → sub 6's measured `limit 0.57112 · prev_sub_bos_threshold (sub 5) · M=none` + sub 7's `limit 0.58105 · parent_bos_threshold · M=none` (CHARTING_SPEC, `fib_hover_limit_text`) |
| A NOTE-5 / B NOTE-2: the tripwire looked up a reversing sub before the counter test — a counter reversal sub with 0 / 2 candidates got a false "cannot be re-derived" mismatch | CODE: the counter test first (`counter = …`; no lookup for a counter sub); pin `test_a_counter_reversal_sub_is_re_derived_without_its_reversing_sub`; mutant M2R1 |
| A NOTE-6: `census_l2_report`'s CL-7 count sees only the moves after the start, not a folded one | noted (the tool's scope; the pinned comparison is unaffected) |
| A NOTE-7: §3's "Started by" column vs the direction-keyed COUNTER row; the formula paragraph omitted "every step mapped and priced" | both |
| B MINOR-1: the fire-time closure's parent CYCLE unpinned (X01 `(S, 0)` survived) | the driver test drives a COUNTER trigger on parent cycle 1 with (0, 0) / (0, 1) steps at distinct prices; X01 killed |
| B MINOR-2: neither R6-2 (a) shape pinned for counter subs (X04 the creating record's parent, X07 its start — survived) | `test_the_tripwire_reads_the_started_by_records_parent_sid` (shape i), `test_the_tripwire_re_derives_a_counter_sub_at_its_started_by_records_start` (shape ii) |
| B MINOR-3: the collapse's M15 key untested across two H1 hours (an M15 gap maps two hours to one LOH; X05 survived — a loud crash, not a wrong value) | `test_two_hours_sharing_one_m15_moment_collapse_too` |
| B MINOR-4: `census_l2_report` MATCH did not check the source | `src_ok` added (the object's and every exported record's `parent_bos_threshold`); the report re-run unchanged |
| B NOTE-1: edited fixtures used impossible (type, direction) pairs (a confluence `first_counter`) | the dispatch test's x−1 path uses `subsequent_confluence`; the two creating `first_counter` records (s2 `Y`, review-pins `y`) → `subsequent_confluence` |
| B NOTE-3: an unknown trigger type no longer raised on a counter sub; two `pytest.raises` without `match` | CODE: `assert trigger_type in _TRIGGER_TYPES` FIRST (the later, now-implied assert removed — an equivalent mutant otherwise); the raises carry `match`; pin `test_an_unknown_trigger_type_raises_on_a_counter_sub_too`; mutant M2R2 |
| B NOTE-4 / NOTE-5: the real-data test pins the confluence subs only as "not counter"; the CL-7 "0" is a real result (two moved subs' breaks after their last move) | none |

**After the fix-up (the final L2 code):** suite **1691 passed** (`test_plan_m_landing_2.py` 34); the review mutants
`mutants_m2_review.py` **6 / 6 KILLED** (`mutants_m2_review_result.txt`); the reference replay re-run — FETCH GATE
PASS, 27 / 27 md5 = `trial_md5.txt`, the three `fig_hash.py` = `trial_fig_hash.txt`, the `[sweep]` lines and "3 subs
re-derived, 0 mismatches" unchanged (157 s wall — shared with the mutants and the census); the census re-run on the
final code — report identical (14 / 14 MATCH, 0 knock-on, 0 tripwire mismatches, CL-7 0).

## 16. Landing 2 — committed and saved (2026-10-12) — PLAN_M COMPLETE

The user's chart review (2026-10-12): **"charts look good. go ahead"**. Source **`10e4c6f`**; `/commit-save` run mode —
replay 48.2 s wall (multi_tf_dual 8.56 s; charts 6.58 / 7.80 / 20.82 s), FETCH GATE PASS (no retry); the save **S_M2
`20261012_023137_10e4c6f`** (27 CSVs + 3 charts; its md5s = `trial_md5.txt`, its `fig_hash.py` = `trial_fig_hash.txt`);
save commit `3ebe857`, trunk `475bbf9`; `LATEST_<branch>` = S_M2. Local, NOT pushed.

**What remains of register §D.1 item 2:** the PARKED follow-up CL-8 — sub 3 cycle 2's IC (2821) vs H1 sid 1 cycle 1's
(732), a residual of sub 3's extra cycle 0 (§4), "looked at after this plan lands" (the user, 2026-10-11).

## 17. The CL-8 follow-up — measured on S_M2 (2026-10-12, session 2) — CLOSED, no change

**The user's read (2026-10-12): "1, keep it — close CL-8 with no change".** The difference is correct: sub 3 really
tracked a cycle (BOS 2639 → CTS 2736, 12-24 → 12-26) that the retroactive H1 sid 1 starts after, and the user's IC
reference rule (`ic_reference_conditions_spec.md`: a cross from cycle 1 takes CTS_0 as a reference, the further-opposite
one wins) applies to it as written. The candidates shown (the user picked 1): (1) keep; (2) a dead previous cycle gives no
CTS reference; (3) a cross inside the window takes the structure reference only; (4) align sub 3's structure (no extra
cycle 0); (5) another rule. Why it closes cheaply: no reference change would reproduce H1's 732 anyway (the shadow below
— the M15 ranks pick 2837). No code, no replay.

Every value extracted from S_M2 (the fib lifecycle + IC candidates CSVs; the M15 candles from the confluence chart's
candlestick trace, the H1 from `_raw.csv`; `plan_m_inputs/cl8_candles.py`). 61.8–80 zone of the shared fib
(0.58424 → 0.574) ≈ 0.58033–0.58219.

- **The same fib, two references.** Both records span BOS 0.58424 (sub 3: 2758, 12-26 15:30; H1: 689, 12-26 15:00) →
  CTS 0.574 (sub 3: 3047, 12-31 15:45; H1: 761, 12-31 15:00). H1 sid 1 cycle 1 = a cross from cycle 0 → no previous
  cycle → the structure reference only (`prev_structure_bos` 0.5736, bound 897) → 12 IC candidates (712–739), one with a
  pattern: **732** (12-30 10:00, 3A.iv, h 0.58118 l 0.58068). Sub 3 cycle 2 = a cross from cycle 1 (cycle 0 was dead
  at cycle 1's establishment → cycle 1 `cross_failed`; the walk for cycle 2 stops at dead cycle 0) → both references:
  `prev_cycle_cts` = CTS_0 0.5818 (2736, 12-26 10:00; bound 2824, 12-29 08:00) and `cross_limit_anchor` 0.5736 (bound
  3588); a sell takes the higher → 0.5818. 130 rows, ONE candidate: **2821** (12-29 07:15, l 0.58182 > 0.5818, before
  2824; no pattern). [Since PLAN_O (2026-10-13): CTS_1 0.57902, bound 2991 → 2837 1A.iv.]
- **The 0.5818 low exists on H1 too:** H1 684 (12-26 10:00) low 0.58180 — five hours BEFORE sid 1's BOS_0 689, so it
  is no cycle of sid 1 (the user's write-up: a cross from cycle 0 has no previous-cycle CTS reference — [superseded
  2026-10-13 by PLAN_O: a cross from cycle 0 reads its target's previous CTS; sid 1's cross 0→1 reads CTS_0 0.57902]).
- **The reference sits INSIDE the zone:** 0.5818 ≈ the 76% retracement of the shared fib — only an IC entirely in the
  zone's top ~4 pips AND before 2824 can pass on sub 3.
- **Shadow (not a replay) — what aligning the reference would pick:** the zone-touching rows of sub 3 cycle 2 that fail
  ONLY the IC reference and carry a pattern: 2817 (2A.ii), **2837** (12-29 11:15, **1A.iv**), 2919 (2A.iv), 2930
  (12-30 10:30, 3C.iv — the M15 counterpart of H1 732, a 3A.iv on H1). By rank (1A first) the pick would be **2837,
  not 2930**: no reference change reproduces 732 — the M15 patterns rank differently from the H1 ones.
