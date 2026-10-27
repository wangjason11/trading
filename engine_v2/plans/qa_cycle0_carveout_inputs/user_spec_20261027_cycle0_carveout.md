# The cycle-0 carve-out revisit (SESSION ORDER item 5b) — the rule, how each cycle 0 is found, the measurements, the options

Register `IDEA_PARKING_LOT.md` §D SESSION ORDER item 5b. A DECIDE item; this file holds the discussion (the user's
words verbatim, the measurements, the options, the decisions as they are made). Read with
`plans/qa_chart_edits_inputs/user_spec_20261024_wvmi_on_chart.md` rounds 6–7 (where the revisit was raised).

## The user's request (verbatim)

2026-10-26 (WVMI round 7): "yes we do need to add line to revisit this. I need to better understand the original carve
out logic. A decision from May was also quite a long time ago and a lot of things have changed since then. From your
description of "a cycle-0 BOS is the swing extreme from before the structure began" that sounds like this was
implemented specifically to cover the case for sid=0 cycle=0 on main TF. In all other cases, cycle0 form as a result of
existing structure logic & triggers. Additionally, when we come to unique subs, there may be many subs that are just
single cycles, and having the carve out logic forces all those to confirm CTS via pullback pattern. Those are my
thoughts for now. We can come back to it in more detail later"

2026-10-27 (the session order): "First, I want to revisit the cycle-0 carve-out that we discussed recently."

## The rule, read from the code (HEAD `3f2a29c`, code = `2dc9e5d`)

`structure/market_structure.py::_maybe_confirm_cts_via_proximity` runs on every candle. Proximity may confirm the CTS
only if ALL hold: `st.cts_phase == "EST_OR_UPD"` (established or updated, not yet confirmed); **`st.cts_cycle_id > 0`
(the carve-out)**; `cycle_gap_ok` = `|st.cts.price − st.bos_threshold| >= min_gap` (the narrow-gap gate — H1 50,
M15 30, M5 15 pips, `zones/zone_proximity.py::DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS`); a BOS inner exists;
`i > st.cts.idx`. Then `_check_sd_proximity_at_candle`: of the BOS inner and the cycle's POI inner, the one closest to
price (the HIGHEST for an up-structure, the LOWEST for a down one); a hit when the candle's low ≤ inner + threshold
(up) / high ≥ inner − threshold (down); threshold = `DEFAULT_PROXIMITY_PIPS` (H1 8, M15 5, M5 3). A hit emits
`CTS_CONFIRMED` (`via: sd_zone_proximity`) and starts a range if none is active; a pullback pattern later in the same
cycle emits `CTS_RECONFIRMED`.

History (git): the carve-out landed 2026-05-11 (`47e7532`, "Main-structure fix + proximity tightening"); the narrow-gap
gate the NEXT day, 2026-05-12 (`fa93e0a`, "Narrow-cycle proximity rules (Rules 1+2+3)"), when the user kept both (Q3,
LANDMINES "Dual CTS Proximity Confirmation — Cycle-0 + Rule 1 Are BOTH Required"). The carve-out was the first fix of
the pathology; the narrow-gap gate is its generic replacement plus the k > 0 cycles. The proximity threshold was 20
pips then (now H1 8). The true-first-breakout cycle-0 redesign that decides how every cycle 0 is found came AFTER both:
designed 2026-06-07, subs 2026-06-08, the main sid 0 2026-06-16, the main reversals 2026-06-20.

The two reasons on file (MARKET_STRUCTURE_SPEC "Cycle-0 carve-out (proximity disabled)", LANDMINES):
1. **Geometric** — the BOS_0–CTS_0 gap is uncontrolled and can be tiny; the proximity buffer then overlaps the CTS
   level and the candle right after `CTS_ESTABLISHED` fires (the May cases). The narrow-gap gate now covers this
   generically, on every cycle.
2. **Semantic** — "BOS_0 = `initial_prior_extreme` is a weaker structural anchor than BOS_k (k > 0) = the prior
   cycle's pullback extreme, even when its gap happens to be wide". This is the ONLY thing the carve-out still adds.

Wording note: the spec says BOS_0 is "the swing extreme that existed *before* the structure began". In the code it is
the extreme between the structure's START and its CTS_0 anchor (`_initial_bos_before_first_cts`: the min low over
`[start_idx, cts_anchor_idx − 1]` for an up-structure, the max high for a down one) — the base of the structure's
first breakout, inside the structure's span. To fix when the spec section is rewritten.

## How each structure's cycle 0 is found (code, 2026-10-27)

Common to all: MS runs in pre-CTS_0 scan mode (`enforce_cts0_new_extreme` + a `bos0_inner` price) from a START;
`structure/true_first_breakout.py::find_true_first_breakout` establishes CTS_0 at the FIRST breakout that (1) closes
past the `bos0_inner`, (2) is a valid breakout pattern re-detected against that threshold, (3) whose full-pattern
extreme is a strict new extreme since the start; BOS_0 = the extreme between the start and CTS_0's anchor, labelled
`source: "initial_prior_extreme"` on every structure. What differs is where the START and the `bos0_inner` come from:

| structure | the start | the `bos0_inner` | resets |
|---|---|---|---|
| main sid 0 (H1, trading open) | `identify_start_scenario_1` over the data (183-day lookback, 50-candle min history) | the ad-hoc BOS_0 zone built at that start (`build_ad_hoc_bos0_reference_zone`) | none — "single-shot" (`structure_engine.compute_structure`) |
| main reversal (H1 sid ≥ 1) | `unified_probe` from the prior sid's most recent CTS anchor to the reversal candle, flipped direction (deterministic only) | the reference zone = that prior CTS (iteration 1), else the ad-hoc BOS_0 at the reset start | retrace resets (PART4 §4.4) |
| sub `first_confluence` | `unified_probe` from the parent BOS anchor PRICE-mapped to M15, bounded at the parent CTS anchor price-mapped; deterministic + the MS-based iterative Phase 2 | the ad-hoc BOS_0 at that M15 candle | retrace resets |
| sub `first_counter` / `subsequent_confluence` / `subsequent_counter` | `unified_probe` from the SIBLING lens's most recent CTS in the trigger's window (fallback: the window's own extreme) to the trigger candle | that sibling CTS zone (iteration 1) | retrace resets |
| sub `reversal` | `unified_probe` from the reversing sub's most recent CTS to its reversal candle | that CTS zone | retrace resets |

So every cycle 0 but the main sid 0 starts from a structural reference (a CTS, a parent BOS) and a probe that has
already validated the start with the retrace resets; the main sid 0 starts where Scenario 1 picks on the data, with no
probe. The user's reading ("built for sid 0 cycle 0 on the main TF") matches the code: the May cases were H1 sid 0
c0 and sid 1 c0, both from before the probes existed in this form.

## Measured (2026-10-27) — three variant replays vs S_wvmi_marks `20261026_230619_2dc9e5d`

`git archive HEAD` into a short `%TEMP%` dir, patched by this folder's scripts; every variant passed the fetch gate
(raw H1 byte-equal), skip grep 33 = the baseline's.
- `c0shadow` (`make_variant_c0.py DIR shadow`): behaviour unchanged (30 / 30 CSVs byte-equal) + a `[c0probe]` line on
  every cycle-0 candle where proximity WOULD hit (every other gate printed). 64.8 s.
- `c0drop` (`make_variant_c0.py DIR drop`): the carve-out removed. 66.2 s. 10 / 30 CSVs differ — M15 only.
- `c0dp` (`c0drop` + `make_variant_probe_tf.py DIR`): also the FC probe's Phase 2 MS on the caller's timeframe (see
  "The probe's H1 pip values" below). 69.3 s. vs `c0drop`: 7 confluence CSVs differ (sub 0 only).

Tables: `c0_table.py SAVE_OR_DEBUG_DIR` (one row per structure's cycle 0); `c0_evdiff.py BASE VARIANT` (per-sub
event diff).

### What the carve-out blocks today (the shadow, every structure's cycle 0)

| structure | dir | BOS_0 → CTS_0 gap at est. → widest before confirm | proximity hits in cycle 0 | blocked by |
|---|---|---|---|---|
| H1 sid 0 c0 (main) | +1 | 31.8 → 243.8 p | 16, from 116 (the candle after est. 115): POI inner 0.56064, 6.2 p under CTS 0.56126 | narrow-gap alone (every hit's gap < 50) |
| H1 sid 1 c0 (reversal) | −1 | 22.9 → 52.2 p | none | — |
| sub 0 c0 (first_confluence, parent (0,0)) | +1 | 16.8 → 144.6 p | from 525; first with gap ≥ 30 at **598** (BOS inner 0.55942) | the carve-out |
| sub 1 c0 (reversal) | −1 | 19.3 → 19.3 p | none | — |
| sub 2 c0 (reversal) | +1 | 16.2 → 107.4 p | none | — |
| sub 3 c0 (reversal) | −1 | 6.4 → 29.8 p | from 2652 (POI 0.58416), gap ≤ 29.8 | narrow-gap alone (0.2 p short of 30) |
| sub 4 c0 (subsequent_confluence, single-cycle) | +1 | 19.5 → 68.7 p | from **3336** (POI 0.57658), gap 68.7 | the carve-out |
| sub 5 c0 (subsequent_counter, single-cycle) | −1 | 14.0 → 99.3 p | none | — |
| sub 6 c0 (first_counter, single-cycle) | +1 | 27.6 → 69.4 p | none | — |
| sub 7 c0 (subsequent_confluence, single-cycle) | −1 | 16.5 → 57.2 p | from 3787 (the candle after est. 3786); first with gap ≥ 30 at **4071** (POI 0.57653) | the carve-out |
| sub 8 c0 (subsequent_counter, both lenses) | +1 | 7.8 → 31.2 p | from 4032 (the candle after est. 4031); first with gap ≥ 30 at **4098** (POI 0.57484) | the carve-out |

(Sub M15 runs are slice-local in the log; absolute = the sub's `starting_idx` − 50 + the logged `i` — checked against
`c0drop`'s confirmations 598 / 4071 / 4098. The FC probe's Phase 2 MS also logs, as `tf=H1` on the full M15 frame,
absolute — see below. The May pathology — a hit on the candle right after CTS_ESTABLISHED — still occurs on H1 sid 0
(116), sub 7 (3787) and sub 8 (4032); the narrow-gap gate blocks all three.)

Correction to the WVMI design file's round-7 facts: the two May cases' "~12-pip BOS–CTS gap" was the POI-inner-to-CTS
distance (LANDMINES' numbers); their BOS_0–CTS_0 gaps are 31.8 p (sid 0) and 22.9 p (sid 1) today — both under H1's
50, so the conclusion (the narrow-gap gate blocks them on its own) holds. Sid 0 c0's case still reproduces (the POI
inner is now 6.2 p under the CTS, at 116); sid 1 c0 has no proximity hit any more.

### Dropping the carve-out (`c0drop`)

H1: no CSV changes. Triggers, the sub list, every sub's `starting_idx` / `start_idx` / end: unchanged. Four subs'
cycle-0 CTS confirms earlier, each by proximity:

| sub | today: CTS_CONFIRMED | without the carve-out | earlier by | what else moves |
|---|---|---|---|---|
| sub 8 (counter + confluence) | 4103, pattern `one_maru_opposite` | 4098, POI 0.57484 | 5 candles | 4103 → `CTS_RECONFIRMED`; the fib's `locked_at` 4103 → 4098; +2 temporary-LP rows per lens (4098, 4101); the WVMI record's final values unchanged |
| sub 7 (single-cycle) | 4157, pattern `one_maru_continuous` | 4071, POI 0.57653 | 86 | 4157 → `CTS_RECONFIRMED`; `locked_at` → 4071; +15 temporary-LP rows (4070 … 4156); final values unchanged |
| sub 4 (single-cycle) | 3489, pattern `one_maru_continuous` | 3336, POI 0.57658 | 153 | 3489 → `CTS_RECONFIRMED`; `locked_at` → 3336; +14 temporary-LP rows (3336 … 3488); final values unchanged |
| sub 0 (5 → 6 cycles) | 802, pattern `one_maru_opposite` | 598, BOS inner 0.55942 | 204 | a NEW cycle 1 at 685 (CTS 0.56738 `one_maru_continuous`, BOS_1 0.55914 anchored 630 → a new BOS zone 0.55914–0.55948); today's cycles 1–4 become 2–5 (every per-cycle row renumbered: fibs, POIs, KL zones, IC candidates 481 → 460 rows, WVMI 4 → 5 records); all of the new structure sits BEFORE the sub's lifecycle start 1020 (both new cycles' WVMI records collapsed) |

The candles (M15 confluence, S_wvmi_marks; proximity = 5 pips):

| sub | idx | time (UTC) | o | h | l | c | wick to the inner |
|---|---|---|---|---|---|---|---|
| 7 (−1; POI inner 0.57653, fires when high ≥ 0.57603) | 4000 | 01-15 14:00 | 0.57300 | 0.57351 | 0.57255 | 0.57346 | (the CTS anchor, 0.57255) |
| | 4070 | 01-16 07:30 | 0.57568 | 0.57589 | 0.57541 | 0.57589 | 6.4 p short |
| | **4071** | 01-16 07:45 | 0.57587 | 0.57638 | 0.57581 | 0.57586 | **1.5 p short — fires** |
| | 4154 | 01-19 04:30 | 0.57732 | 0.57756 | 0.57723 | 0.57755 | 10.3 p past (the pattern's first candle) |
| | 4157 | 01-19 05:15 | 0.57765 | 0.57798 | 0.57760 | 0.57785 | 14.5 p past — today's confirmation (high above the POI's top 0.57705 and the BOS inner 0.57758) |
| 4 (+1; POI inner 0.57658, fires when low ≤ 0.57708) | 3304 | 01-06 08:00 | 0.58044 | 0.58105 | 0.58036 | 0.58050 | (the CTS anchor, 0.58105) |
| | 3334 | 01-06 15:30 | 0.57808 | 0.57823 | 0.57718 | 0.57764 | 6.0 p short |
| | **3336** | 01-06 16:00 | 0.57776 | 0.57791 | 0.57688 | 0.57742 | **3.0 p short — fires** |
| | 3489 | 01-08 06:15 | 0.57581 | 0.57582 | 0.57513 | 0.57516 | 14.5 p past — today's confirmation |
| 0 (+1; BOS inner 0.55942, fires when low ≤ 0.55992) | 589 | 11-25 01:15 | 0.56178 | 0.56218 | 0.56150 | 0.56152 | (the CTS anchor then, 0.56218) |
| | 597 | 11-25 03:15 | 0.56132 | 0.56134 | 0.56080 | 0.56082 | 13.8 p short |
| | **598** | 11-25 03:30 | 0.56081 | 0.56084 | 0.55986 | 0.56016 | **4.4 p short — fires** |
| | 802 | 11-27 06:30 | 0.57171 | 0.57184 | 0.57154 | 0.57178 | today's confirmation (the CTS had run to 0.5732) |
| 8 (+1; POI inner 0.57484, fires when low ≤ 0.57534) | 4086 | 01-16 11:30 | 0.57667 | 0.57674 | 0.57645 | 0.57654 | (the CTS anchor, 0.57674) |
| | **4098** | 01-16 14:30 | 0.57552 | 0.57554 | 0.57514 | 0.57522 | **3.0 p short — fires** |
| | 4103 | 01-16 15:45 | 0.57446 | 0.57502 | 0.57444 | 0.57460 | 4.0 p past — today's confirmation |

### The probe's H1 pip values (a deferred cleanup the carve-out keeps unreachable)

The FC probe's iterative Phase 2 runs MS through `structure_engine._make_market_structure` WITHOUT `timeframe`, so on
the M15 frame it uses the H1 proximity 8 p and gap 50 p (the helper's docstring: "Parity carve-out … proximity_pips
for probes is a deferred cleanup"; `unified_probe._run_phase2` says the probe's result "never [reads] a proximity
confirmation"). With the carve-out, cycle 0 never confirms by proximity, so the probe's cycle 0 matches the sub's own
run. Without it, they can disagree: in `c0drop` the probe's MS (gap 50) still has its 2nd CTS at 1020 while sub 0's
own MS (gap 30) has it at 685 — the sub's lifecycle start stays 1020. In `c0dp` (the probe on M15 values): sub 0's
`start_idx` 1020 → 685 (`probe_finalize_idx` 685, `second_cts_reached`), its new cycle 1 record no longer collapsed
(+11 per-LP rows); nothing else moves.

## Options (put to the user 2026-10-27)

- **A. Keep the carve-out** (today).
- **B. Drop it everywhere** — the narrow-gap gate alone guards proximity on every cycle. On the window = `c0drop`.
- **C. Keep it only on the main sid 0** (the trading-open structure, the one cycle 0 with no structural reference and
  no probe); drop it on main reversals and every sub. On the window = B (both H1 cycle 0s are unchanged either way).
  Needs an explicit flag from `compute_structure` (sub MS runs are `structure_id=0` too).
- **D. Keep it on the main H1 (sid 0 + reversals), drop it on subs.** On the window = B.
- Not an option: "single-cycle subs only" — a sub's cycle count is known only when it ends.
- With B / C / D, a second question: the FC probe's Phase 2 pip values — leave them (sub 0's lifecycle start stays
  1020 while its own 2nd CTS is 685) or give the probe the caller's timeframe (`c0dp`: 1020 → 685). Its own landing
  either way (one cause per `/compare`).

## Round 1 — the user's answers (2026-10-27, verbatim)

> [carve-out] C: main sid 0 only (Recommended)
>
> [the probe's pip values] I think the right behavior would be sub 0's lifecycle start stays at 1020. However, it's own
> 2nd CTS should still be consistent. I think this is a result of sub_0 will always be the FC of sid_0 cycle_0.
> Therefore, I think what we actually can/should do is keep the carve-out on both the main sid 0 and the sub_id 0.
> What do you think about that?

Then, to the author's check of whether `first_confluence` fires only on main sid 0 cycle 0: "yes it can fire for other
parent cycles too. what I'm saying is that sub_0 would effectively always be fired via FC".

Checked (2026-10-27): `first_confluence_trigger.detect_first_confluence_triggers` emits one trigger per parent
BOS_CONFIRMED — every cycle of every sid, cycle 0 included (its docstring's "var 1 doesn't fire for cycle 0" is
stale: cycle 0 has a BOS_CONFIRMED, `initial_prior_extreme`, at CTS_0's moment). On the window: FC (0,0) at 463 → sub
0; FC (0,1) at 2611 → a record on sub 2; FC (1,2) at 3611 → a record on sub 5; FC (1,0) / (1,1) unresolved
(`collapsed_parent_cycle`). FC (0,0) is the earliest trigger the main can make (CTS_0's moment; first_counter and the
subsequent triggers need a proximity trigger after it), so the first unique sub is the one it starts — unless its
probe fails. Sub 0's probe input is the main sid 0's BOS_0 anchor (H1 96 → M15 385): its cycle 0 sits on the same
trading-open BOS_0 the main sid 0 keeps the carve-out for.

The author agreed, with two refinements: key the rule on what the sub IS (the sub started by the first_confluence of
the main sid 0 cycle 0), not on `sub_id == 0` (sub ids are an output numbering — renumbered before, PLAN_R — and a
failed / pending FC (0,0) probe would make another sub "sub 0"); the FC (0,0) probe's own MS keeps the carve-out too
(so it and sub 0 agree on the 2nd CTS, 1020), every other FC probe drops it like its sub. Predicted (= `c0drop` minus
sub 0, whose changes reached no other structure there): only subs 4, 7, 8 change.

## Round 2 — the probe's pip values (2026-10-27, verbatim)

> My feeling is saying that we shouldn't make the change on this one. But do you have more information on the comment
> you mentioned? Are there any additional notes or documentations around whhy we chose those original parameters? And
> any information on rationale for potentially wanting to change this?

Found (git + docs, 2026-10-27):
- **Origin — not a parameter choice.** `df11996` (2026-05-05, Part 4 §13.5.b, the resolver-injection refactor, built
  byte-identical): the probe call sites of the time (the Exception 2 probes, the Scenario 3 Phase 1 / 2 probes — all
  deleted since) had always built MS without `timeframe`, so they got the H1 default (20 pips then). The refactor kept
  it ("Parity carve-out (matches Step 1 baseline)"; "Spec §4.4 explicitly listed only the probe-RESET threshold for
  TF-aware lookup; proximity_pips for probes is a deferred cleanup"). The narrow-gap gap (`fa93e0a`, 2026-05-12)
  reached the probe the same way (`setdefault` by the absent timeframe → H1's 50). `unified_probe`'s Phase 2 is the
  one site left.
- **Registered 2026-10-19** (PLAN_R Q&A rounds 5–7, `qa_opp_sd_zones_inputs/qa_findings.md`; the user: "What is this?
  And why do we need it?", then "Ok sounds good. We can add it"): register §E "The start-candle probe's market
  structure runs at the H1 proximity values on M15" — INERT BY DESIGN because the probe reads cycle 0's CTS_CONFIRMED
  and cycle 0 cannot be proximity-confirmed; **TRIGGER: the probe ever reads a proximity-confirmed CTS → pass the
  sub's `timeframe`.**
- **Round 1's rule fires that trigger** for every FC probe except the one that starts sub 0. The probe reads cycle 0's
  CTS_CONFIRMED for its retrace window (the first CTS anchor) and for its `second_cts_reached` finalize (the 2nd CTS
  needs cycle 0 confirmed) = the sub's lifecycle start. A probe and its sub can then disagree both ways: the sub
  confirms by proximity and the probe does not (a gap of 30–50 pips — sub 0 in `c0drop`: probe 2nd CTS 1020, own 685),
  or the probe confirms and the sub does not (a wick 5–8 pips short with a gap ≥ 50 — the probe's 2nd CTS EARLIER than
  any the sub has).
- **A second coupling on the same premise:** the probe's MS reads its POI inner with NO cross-fib limit
  (`CrossFibLimit(None, "none")`, PLAN_I §9.2 F7: "the probe's result does not depend on the POI read … cycle 0's
  CTS_CONFIRMED — never by proximity"), while a sub's cycle-0 fib carries its limit when the parent cycle is ≥ 1
  (fib lifecycle: sub 4 `parent_bos_threshold`, sub 7 `cross_limit_anchor`; FC on c ≥ 1: `parent_cts_threshold`). So
  aligning the pips alone would not make a probe read what its sub reads.
- On the window: no effect (the drop + probe-fix replay moved only sub 0; the other FC triggers, (0,1) and (1,2),
  attach to subs 2 and 5 unchanged).

## Round 3 — how the 8 and 50 are used (2026-10-27, verbatim)

> I still need some more information. Can you explain in more detail how the 8 & 50 values are used?

Traced (code, 2026-10-27):
- **Where MS reads them — the proximity confirmation path only.** `_make_market_structure` sets
  `proximity_pips` / `min_gap_pips` from the timeframe's table (absent → H1: 8 / 50); MS turns them into
  `_proximity_threshold` (the buffer, read once: `_check_proximity_at_candle` → a hit when the low ≤ inner + 8 pips, up
  / the high ≥ inner − 8, down) and `_min_gap_threshold` (read once: `_maybe_confirm_cts_via_proximity`'s
  `cycle_gap_ok`, |CTS − BOS threshold| ≥ 50); `proximity_pips` is also written into the proximity CTS_CONFIRMED's
  meta. Nothing else in MS reads either value (grep `market_structure.py`).
- **What the FC probe's Phase 2 reads from its MS run** (`unified_probe._run_phase2`): the CTS_ESTABLISHED events,
  cycle 0's CTS_CONFIRMED, a reversal. Cycle 0's CTS_CONFIRMED is used three ways: (1) the retrace window
  `[CTS_0 moment + 1, that confirmation's CTS anchor − 1]` (else to `probe_end_idx`) → the deepest retrace candidate →
  the reset test → the start can move; (2) a `no_retrace` finalize = that confirmation's candle; (3) the 2nd
  CTS_ESTABLISHED can only come after it (MS opens a new cycle only from a confirmed one) → `second_cts_reached` =
  the 2nd CTS's moment, and the early stop (Plan B). The finalize = the sub's lifecycle start. Cycle 1's own
  confirmation is never read (the run stops at the first quiescent point after the 2nd CTS).
- So today both values decide nothing: cycle 0 cannot confirm by proximity (the carve-out), and nothing past cycle 0's
  confirmation is read.
- **Sub 0's probe vs sub 0 itself** (`c0shadow`, the carve-out's would-be hits): the probe (8 / 50) has 57 hit
  candles in cycle 0, from 465 (gap 25.2 p); the sub (5 / 30) has 34, from 525 (28.4 p) — the wider buffer counts
  candles the sub does not (465, 515–517). Both stop at 644 / 643 with the gap at 34.4 p (CTS 0.56218, anchored 589;
  BOS 0.55874): ≥ the sub's 30 → the sub fires at 598 without the carve-out; < the probe's 50 → the probe never does
  (by the time the gap passes 50, price is far above both inners — no hit until the pattern at 802).

## Round 4 — the decision on the probe's values (2026-10-27, verbatim)

> I see. I think I got a bit confused and thought that the 8 pips was being used for determining whether the probe
> resets. For that we are using a different value correct and this has nothing to do with that? Furthermore, you are
> saying that currently the M15 MS is using the 5/30 while the M15 probe uses 8/50? If that is the case, then I do
> think that we need to update to 5/30 and be consistent.

Confirmed: the probe's reset uses its own values, already on the caller's timeframe (`unified_probe.unified_probe`:
`DEFAULT_PROBE_RESET_PIPS` H1 4 / M15 3 / M5 2 for the reset's closeness test, `DEFAULT_PROBE_RESET_WICK` 16 / 12 / 8
for its wick cap — an M15 probe resets at 3 / 12); the 8 / 50 are never read by the reset. And yes: in the replay log
the sub's own MS runs as `tf=M15` (5 / 30), the FC probe's Phase 2 MS as `tf=H1` (8 / 50) on the same M15 candles.

**Decided: the FC probe's MS uses the caller's timeframe values (M15 5 / 30), consistent with its sub.**

## Round 5 — the cross-fib limit + the landing order (2026-10-27, verbatim)

> [the probe's cross-fib limit] yes let's include. In general, my impression is probe effectively should function as a
> full MS run, and when the probe ends, it allow us to copy over probe results as MS, and continue on without needing
> to re-run new MS on those same probe candles. To enable that, the probe should and need to be the same as final MS.
> That's my understanding. However, if that is not the case, let's also not force anything right now.
>
> [order] Probe first (Recommended)

Decided: **landing 1 = the FC probe's MS reads what its sub reads** (the caller's timeframe values AND the sub's
cross-fib limit); **landing 2 = the carve-out rule** (round 1).

On the user's impression (the design as built, memory `project_true_first_breakout_cycle0.md` "UPDATE 2026-06-07"):
the probe hands MS a DECISION — the start + the BOS_0 inner (+ CTS_0's moment as a sanity assert) — not events; the
sub's MS then runs from that start over the same candles and produces every event itself ("scan-from-start", decided
2026-06-07 over "seed-and-resume" = copying the probe's state and continuing): one MS path for the breakout-found and
the not-found cases, agreement by construction, the re-run's cost negligible; seed-and-resume was kept as a possible
later optimization. The user's principle — the probe must behave exactly like the final MS — is what makes that
agreement hold, and landing 1 restores it for the FC probe's MS. Not forced now (the user): no move to copying the
probe's results. One difference stays by design: a probe's MS is BOUNDED at `probe_end_idx` (truncation semantics,
Plan A — it reads nothing past its bound), so near its bound it can differ from the unbounded run.

**The plan: `engine_v2/plans/PLAN_T_probe_parity_and_cycle0_carveout.md`** (2026-10-27) — T1 the probe, T2 the
carve-out; their measured predictions (variants `c0T1`, `c0T2`, `make_variant_rule_c_sub0.py` in this folder) live
there (§2.6, §3.3).
