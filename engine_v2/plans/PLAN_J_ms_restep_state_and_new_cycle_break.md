# Plan J — MarketStructure: a re-stepped candle sees the state as of itself; a new cycle must break its threshold; the BOS window ends at the CTS anchor

**Status: round 1 ANSWERED 2026-10-09 (§7: J1a, J2b; J3 per R9-2) — the build next (§6). No engine code changed yet.** Opened 2026-10-08 (session 6 of the cross-cycle fib
revamp) from PLAN_I's round 9: R9-1 (b) — "an MS fix for F16 as its own plan (a back-filled candle's re-step sees the
state as of that candle — the pullback at 13:30 confirms the CTS known at 13:30)"; R9-2 (a) — "(i) a breakout
establishing cycle T must break cycle T−1's confirmed `cts_threshold` as it stands at the apply (R3-1); (ii) the
cycle-T BOS window ends at the CTS anchor; the S2 / S2b raises stay as guards". The findings: PLAN_I §9.2 F16 / F17.
Register: `IDEA_PARKING_LOT.md` §D.1 item 2.

**Baseline:** HEAD `515c449` (PLAN_I S5b). Its replay outputs == `%TEMP%/pi/S5` (27 CSVs + 3 figures + run.log; S5b
is byte-identical to S5). Save: S_F1 `20261006_224610_42c137c` (= `LATEST_<branch>`; no save since).
**Tools (`plan_j_inputs/`):** `variant_patch_j.py` (the variant patches on a `git archive` of `515c449`),
`replay_variant_j.py` (the window replay through a tree; dumps `PJ_LOG`), `search_tree.sh` + `search_report.py` (the
session-4 random-tail corpus through a tree, summarised against the base tree), `trial_detail.py` (one search trial,
events + candles), `core_sig.py` + `core_report.py` (an order-free structural signature per trial), `evdiff.py`
(structure-events CSV deltas), `j2c_check.py` / `j2_after.py` (J2's voids). Trees / work dirs: `%TEMP%/pj/`. Window CSV / figure diffs: `plan_e_inputs/review_scripts/
cmp_fig_full.py` (`PYTHONPATH=.`) against `%TEMP%/pi/S5`.

**Three causes → three stages, one `/compare` each (§6).**

---

## 1. The causes

### J1 — F16: a range back-fill's state leaks into the re-stepped candles

**Mechanism (`structure/market_structure.py`).** Two sites back-fill a range candidate offline, then step the same
candles AGAIN:
- `_step_anchor`'s no-winner branch: candle `i` is a range candidate confirmed at `c` → the frozen per-candle step
  (`_replay_step_no_patterns(k, freeze_range=True)`) runs over `[i, min(c, D))`, `_finalize_range_candidate_offline(i)`
  starts the range (seeded with candle `i`'s high / low and `st.cts`'s price; `RANGE_STARTED` stamped at `c` with the
  CTS in its meta), `i` is re-stepped, and the step returns `i + 1` — so `i + 1 … c − 1` are stepped again as anchors;
- `_post_apply_range_check` (after a breakout applies at `a`): the same over `[a, min(c, a + K))`, then `_step_anchor`
  returns `a + 1`.

The frozen step does everything but the range update: the raw CTS update, the sd-zone proximity confirmation, the BOS
barrier (a probe moves `bos_threshold`; a close-break opens a watch), the pending reversal. So when anchor `k` in
`(i, c)` is re-stepped — its pattern detection included — MS already holds the state as of `c − 1`. A reversal applied
inside the back-fill is terminal and stops the step there (MARKET_STRUCTURE_SPEC "Reversal inside a back-fill") — that
path re-steps nothing and is not affected.

**The real example (PLAN_I §5 R9-1; NZD_USD M15 2025-03-24, a bearish structure, `f14_slice.py M15 5515 173 11 -1`;
slice candle i = 10:30 UTC + 15 min × (i − 11)):**

| candle | UTC | low | today | J1a (§2) |
|---|---|---|---|---|
| 19 | 12:30 | 0.57236 | the CTS (pattern path) before candle 20 | same |
| 20 | 12:45 | 0.57234 | range candidate, confirmed at 25 (14:00); back-fill `[20, 25)`: raw CTS → 20, 21, **24** | raw CTS → 20 in its re-step |
| 21 | 13:00 | 0.57230 | (re-step: nothing new) | raw CTS → 21 in its re-step |
| 22 | 13:15 | 0.57240 | the pullback `one_maru_continuous`'s first candle | same |
| 23 | 13:30 | 0.57268 | `CTS_CONFIRMED` anchor **24** (0.57191) — candle 24 has not printed | `CTS_CONFIRMED` anchor **21** (0.5723) |
| 24 | 13:45 | 0.57191 | (raw update already applied in the back-fill) | post-confirmation: `RANGE_UPDATED` + `CTS_THRESHOLD_UPDATED` 0.57191 |
| 25 | 14:00 | 0.57182 | `RANGE_STARTED` (start 20, meta CTS 24); `CTS_THRESHOLD_UPDATED` 0.57182 | `RANGE_STARTED` (start 20, meta CTS **19**); `CTS_THRESHOLD_UPDATED` 0.57182 |

Today MS emits `CTS_UPDATED`(24) before `CTS_CONFIRMED`(23) — S5's end-of-run order tripwire raises on this slice;
under J1a it does not, and the drawn fib, MS's fib, the KL CTS zone and the next cycle's walk all read CTS 21.

**What leaks on the reference window (`census` tree, every MS run of the replay — main, probe Phase 2, geometry
builds):** 54 non-terminal range back-fills (no-winner: M15 28, H1 15; post-apply: M15 8, H1 3); **25 moved the CTS**
(1–5 raw `CTS_UPDATED`s inside; 1 of them also refreshed the POI inners); **0** proximity confirmations, **0**
`bos_threshold` moves, **0** watches opened inside one. F16 itself (a confirmation whose anchor lies after it): 0 on
the window. The `tf` label of the census is MS's `timeframe` argument.

**The design note.** MARKET_STRUCTURE_SPEC "Look-ahead windows without rewinds" says the candles between a range
candidate and its decision "are processed with the thresholds known at the decision". The range's ACTIVATION from its
candidate candle is that documented look-ahead (a range confirmed at `c` exists from `i`). Its BOUNDS today are partly
decision-time: the sd-side bound is seeded with the CTS the back-fill reached (`st.cts` at finalize), the other side
with candle `i`'s extreme, then both grow with the re-steps. R9-1 is about the state the re-steps read; where the
range's bounds fall is J1-Q1 (§5).

### J2 — F17 (i): a new cycle established without breaking the previous cycle's threshold

**Mechanism.** `_step_anchor` picks the winner at its anchor with the detection threshold of THAT moment (the range's
breakout bound, or **None** when no range is active — `check_break` then always passes). The winner's frozen back-fill
`[anchor, apply)` runs in candle order and may CONFIRM the current cycle (a proximity hit) and set `cts_threshold`. At
the apply `_apply_pattern_at_apply_idx` sees `cts_phase == "CONFIRMED"` and ESTABLISHES cycle T + 1 — with a pattern
never tested against that threshold. The `cts_frozen` WARNING (PLAN_I S2, D3) prints exactly when the detection
threshold differs from `cts_threshold` at an establishment; 0 on the window (19 / 19 equal, PLAN_I §9.2 F12).

**The example (WAVES seed 413 trial 1074 — synthetic M15 candles, sd −1; `trial_detail.py waves 413 1074 40 50`):**

| candle | high | low | close | today |
|---|---|---|---|---|
| 43 | 0.56773 | 0.56461 | 0.5655 | cycle 1 ESTABLISHED (CTS 43, 0.56461) |
| 44 | 0.56564 | 0.5632 | 0.56338 | raw CTS → 44; a `one_maru_continuous` breakout anchored here (no range → threshold None) |
| 45 | 0.56361 | 0.55877 | 0.55897 | raw CTS → 45 |
| 46 | 0.56285 | 0.55791 | 0.56188 | raw CTS → 46 (0.55791) |
| 47 | 0.5785 | 0.56133 | 0.56209 | cycle 1 CONFIRMED by proximity (anchor 46); range started (`cts_threshold` 0.55791); `BOS_THRESHOLD_UPDATED` 0.5785 |
| 48 | 0.56246 | 0.56115 | 0.56145 | — |
| 49 | 0.56147 | 0.55799 | 0.55809 | the breakout applies: cycle 2 ESTABLISHED, CTS anchor **46** (cycle 1's own CTS candle), BOS anchor **47** (0.5785); `cts_frozen` 0.55791 ≠ the detection threshold None (WARNING) |

No candle of 44–49 closed below 0.55791. MS's order tripwire raises on this run (shape (2)); the downstream dispatch
would too (BOS anchor 47 after CTS anchor 46 — PLAN_I S2b). **J2 (§2):** at 49 the breakout is re-detected at its
anchor 44 against 0.55791 — candle 44 closed at 0.56338, not below → VOID: no cycle 2, no tripwire, no skip.

### J3 — F17 (ii): the BOS of a new cycle after its CTS anchor

**Mechanism.** `_select_bos_on_breakout(apply_idx)` takes the deepest pullback extreme in `[last pullback apply |
cts_confirmed_idx, apply]` — a window that runs PAST the new CTS anchor when the breakout's extreme is not its apply
candle, so the BOS can lie after the CTS. MS asserts only BOS anchor ≤ its moment; the downstream dispatch raises on BOS
after CTS (S2b); MS's order tripwire does not (the establishment and its BOS share their moment).

**The example (WAVES seed 301 trial 572 — synthetic H1 candles, sd −1):**

| candle | high | low | close | today |
|---|---|---|---|---|
| 39 | 0.59943 | 0.58326 | 0.58392 | the range's pullback side expands (`STATE_CHANGED` → pullback, `replay_expand_pullback_side`; `CTS_THRESHOLD_UPDATED` 0.58326) — the highest high of 7–43 |
| 40 | 0.58417 | 0.58233 | 0.58252 | the breakout `one_maru_continuous` anchored here |
| 41 | 0.58288 | 0.57881 | 0.57976 | |
| 42 | 0.58248 | 0.57924 | 0.57961 | |
| 43 | 0.58013 | 0.57645 | 0.57916 | the breakout's extreme → cycle 1's CTS anchor (0.57645) |
| 44 | 0.60401 | 0.57879 | 0.57981 | a 250-pip synthetic wick (`BOS_THRESHOLD_UPDATED` 0.60401) |
| 45 | 0.58018 | 0.5771 | 0.57748 | the breakout applies: cycle 1 ESTABLISHED; BOS window `[7, 45]` (cycle 0's confirming pullback applied at 7) → BOS anchor **44** (0.60401), after the CTS anchor 43 |

**J3 (§2):** the window `[7, 43]` → BOS anchor 39 (0.59943); the dispatch skip is gone. (Window start measured with a
spy on `_select_bos_on_breakout`: `last_pullback_pat_apply_idx` 7.)

---

## 2. Variants (`variant_patch_j.py`; the reference window replayed through each tree, compared with `%TEMP%/pi/S5`)

| Variant | Change | Window CSVs | Window figures | Real F16 slice |
|---|---|---|---|---|
| `census` | instrumentation only | 27 / 27 equal | 3 / 3 equal | — |
| **J1a** restore | after a non-terminal range back-fill (both sites) the state, the events it emitted, the embedded FibTracker and the output rows are restored; the range starts with the CTS as of its candidate candle; the re-steps process the candles in order | **4 differ** (below) | **H1: 2 traces** (`range:top:sid0`, `range:bottom:sid0`) **+ layout shapes**; the two M15 figures equal | no tripwire; CONFIRMED anchor 21; `CTS_THRESHOLD_UPDATED` 0.57191 at 24 |
| J1b CTS-only | no raw CTS update inside a range back-fill (both sites); every other effect kept | == J1a on the window (only the CTS leaks there) | == J1a | == J1a |
| J1c restore, decision seed | J1a, but the range seeded (and `RANGE_STARTED`'s meta CTS stamped) with the CTS the back-fill reached — today's seed | **4 differ: event ROW ORDER only** (same multiset: H1 12, confluence 67, counter 23 positions) + H1 `final.csv` 6 cells | 3 / 3 equal | no tripwire; CONFIRMED anchor 21; but **`CTS_THRESHOLD_UPDATED` 0.57191 at 23 (13:30)** — candle 24's low, not printed yet |
| **J2** re-validate | a cycle ≥ 1 breakout whose detection threshold ≠ `cts_threshold` at its apply is re-detected at its anchor against it; unless the same pattern (name, apply) qualifies it is VOID (no event, no range reset, no state change; the step continues at apply + 1) | 27 / 27 equal (0 re-validations) | 3 / 3 equal | — (413 / 1074 voided, above) |
| **J3** BOS window | the cycle ≥ 1 BOS window ends at the new CTS anchor | 27 / 27 equal (0 window-order swaps) | 3 / 3 equal | — (301 / 572 BOS 39, above) |
| J2 + J3 | both | 27 / 27 equal | 3 / 3 equal | — |

**J1a on the window — every delta** (J1b identical):
- H1 `final.csv`: 8 cells in 5 rows — row 310 (2025-12-03 20:00) `cts_idx` 311 → 309, `cts_price` 0.57774 → 0.57766,
  `range_hi` / `breakout_th` 0.57774 → 0.57766; `cts_event` "" → `CTS_UPDATED` on rows 173, 196, 311, 903 (the raw
  updates now happen in those candles' own steps, which write the marker).
- `structure_events` (H1 328 → 329 rows; confluence 1321 → 1346; counter 476 → 481): every non-terminal back-fill that
  moved the CTS now starts its range with the CTS as of the candidate candle — `RANGE_STARTED`'s meta
  `cts_anchor_idx` / `cts_price` (and its seeded bound) change (`RANGE_STARTED` rows removed / added: H1 4 / 4,
  confluence 16 / 17 — the 17th the new range below, counter 5 / 5), the re-steps add `RANGE_UPDATED`s as the later
  candles widen it (added / removed: H1 1 / 0, confluence 27 / 6, counter 6 / 1), and the raw `CTS_UPDATED`s are emitted
  after `RANGE_STARTED` (row order). ONE behaviour change (confluence: + `RANGE_RESET`, + 2 `STATE_CHANGED`, one
  `CTS_UPDATED` raw → pattern path) — the subsequent_confluence sub, M15 2026-01-13 (sd −1):

  | M15 | UTC | high | low | close | today | J1a |
  |---|---|---|---|---|---|---|
  | 3813 | 15:15 | 0.57498 | 0.57422 | 0.5744 | CTS (pattern path) 0.57422 | same |
  | 3814 | 15:30 | 0.5746 | 0.57424 | 0.57438 | range candidate, confirmed at 3819 (16:45); back-fill raw CTS → 3816, **3817** | — |
  | 3815 | 15:45 | 0.57508 | 0.57436 | 0.57491 | range low already 0.57351 (seeded from 3817) | range low 0.57422 |
  | 3816 | 16:00 | 0.57496 | 0.5738 | 0.57385 | a `one_maru_continuous` anchored here fails: close 0.57385 not below 0.57351 | it qualifies: close 0.57385 below 0.57422 |
  | 3817 | 16:15 | 0.57391 | 0.57351 | 0.57378 | — | the breakout applies: `RANGE_RESET`, `CTS_UPDATED` 0.57351 (pattern path), BREAKOUT |
  | 3818 | 16:30 | 0.5743 | 0.57372 | 0.57418 | — | a new range candidate, confirmed at 3821 (17:15) |
  | 3819 | 16:45 | 0.57452 | 0.57418 | 0.5745 | `RANGE_STARTED` (start 3814, meta CTS 3817) | — |

  The CTS is the same candle and price either way (3817, 0.57351): KL / POI / fib / WVMI CSVs are unchanged.
- KL, POI, fib, IP, WVMI, triggers, subs CSVs: equal (23 / 27 CSVs md5-equal).

**The test suite through the trees** (full suite, OANDA smoke deselected): J1a and J1c each fail the SAME 9 (1416
pass): the F16 raise pin (`test_plan_i_s5_ms_reads_the_fib::test_the_f16_shape_raises_at_the_end_of_the_run` — the
shape no longer exists), `test_bounded_structure::…::test_first_segment_events_match` and
`test_meta_key_renames::test_a_real_run_emits_only_the_new_keys[±1]` (event ORDER / the proximity range below), and
5 tests on `_make_watch_over_second_cts_data` (`test_ms_stop_after_cts::TestQuiescence` ×4,
`test_unified_probe::TestFinalizeIdxTable::test_phase2_second_cts_reached_is_the_moment_not_the_anchor`). That fixture
(sd +1): cycle 1's breakout applies at 10; candle 10 is a range candidate confirmed at 12, so `_post_apply_range_check`
back-fills `[10, 12)` — candle 11 there confirms cycle 1 by proximity (+ its own proximity range), close-breaks the BOS
0.6088 (a watch, the reversal applies at 14) — and 11 is then re-stepped. Under J1a the watch at 11 and the reversal at
14 are unchanged, but (i) the post-apply range (start 10, confirmed 12) is the active range when 11 is stepped, so the
proximity confirmation creates no range of its own, and (ii) **Plan B's early stop** (`stop_after_cts_established=2`,
the probe's Phase 2) is checked between candles 10 and 11 with the state as of 10 — no watch yet — so it stops at 11;
today it saw candle 11's watch through the back-fill and waited for the reversal. Another look-ahead R9-1's rule
removes; it moves the probe's Phase-2 stop wherever a post-apply range back-fill of the 2nd establishment spans a
watch-opening candle (0 on the window: triggers / subs CSVs unchanged).

**H1 example of the bound difference (2025-12-03, sid 0, sd +1):** candle 309 (19:00, high 0.57766) is the CTS; 310
(20:00, high 0.57766, low 0.57732) a range candidate confirmed at 313 (23:00); 311 (21:00) a new high 0.57774.
Today row 310 shows CTS 311 and the range top 0.57774 — 21:00's high, at 20:00. J1a: CTS 309, top 0.57766 at 310;
311 widens the range (`RANGE_UPDATED` 0.57774) and moves the CTS. J1c: CTS 309, top 0.57774 at 310.

---

## 3. The random-tail corpus

The session-4 corpus (PLAN_I §9.10): WAVES-B seeds 401–424 × 2,500, WAVES 301–312 × 600, REAL 201–212 × 400 (72,000
trials), through the base tree and each variant tree (`search_tree.sh`), summarised by `search_report.py`. `trip` =
MS's end-of-run order tripwire (F16 and F17 (i)); `skip` = a CTS_ESTABLISHED whose BOS sorts after it (the downstream
dispatch raises on it since S2b — F17 (i) and (ii)); `sig` = every event in emission order + every MS output column
(so a re-ordered emission or an as-of debug column counts). The base reproduces session 4 exactly.

| Tree | errors | trip (WB / W / R) | skip trials (WB / W / R) | `sig` ≠ base (WB / W / R) | reversal list ≠ base |
|---|---|---|---|---|---|
| base `515c449` | 0 | 42 / 2 / 3 | 10 / 3 / 0 | — | — |
| J1a | 0 | **6** / 0 / 0 (the 6 = F17 (i)) | 10 / 3 / 0 | 32,063 / 3,847 / 3,597 | 3 / 0 / 2 |
| J2 | 0 | 36 / 2 / 3 | **4** / 3 / 0 | 104 / 8 / 0 (= the 112 voids) | 2 / 0 / 0 |
| J3 | 0 | 42 / 2 / 3 | 6 / **0** / 0 | 5 / 3 / 0 | 0 |
| **J1a + J2 + J3** | 0 | **0 / 0 / 0** | **0 / 0 / 0** | 32,094 / 3,850 / 3,597 | 5 / 0 / 2 |
| J1c, J1b, J2b | (running — appended when in) | | | | |

**A structural signature** (`core_sig.py` + `core_report.py`: CTS_ESTABLISHED / BOS_CONFIRMED / CTS_CONFIRMED with
their anchors and prices + the reversal, order-free; and the order-free multiset of every event) on a subset — WAVES-B
401–402 + WAVES 301–304 (7,400 trials), REAL 201–206 (2,400):

| Tree | event multiset ≠ base (WAVES / REAL) | structure ≠ base (WAVES / REAL) | of which |
|---|---|---|---|
| J1a | 3,898 / 1,783 | 72 / 38 | WAVES: 57 a confirmation only, 15 the cycle path (EST + BOS + CONF); REAL: 25 confirmation only, 11 cycle path, 2 + the reversal |
| J1c | 372 / 7 | 28 / 2 | confirmations only |

Both remove every F16 trip of the subset (WAVES 7 → 0, REAL 2 → 0). A confirmation that moves WITHOUT a trip is the
second leak kind — a proximity confirmation inside a back-fill seen by an earlier re-stepped candle: e.g. WAVES 301 / 11
(H1, sd −1) — today cycle 1 is CONFIRMED by proximity at 19 (inside a range back-fill), and the pullback that applies at
18 when 18 is re-stepped only RE-confirms; with the restore the pullback confirms cycle 1 at 18 (anchor 16, 0.58182
either way). J1b (CTS only) cannot fix this kind.

**J2's voids (the 112 trials; `plan_j_inputs/j2c_check.py`, `j2_after.py`):** every one had detection threshold None (a
breakout picked while no range was active, its cycle confirmed during its own back-fill). Base printed 113 `cts_frozen`
WARNINGs; J2 0 (112 voided; the 113th establishment no longer arises). The apply candle CLOSES beyond `cts_threshold`
in **104** of the 112, not in 8 (413 / 1074 is one of the 8). After the void, cycle T (the cycle the base establishes
at the void's apply) is established:
- **J2** (continue at apply + 1): in 65 trials, median 6 candles later (2 … 55); never within the trial in 47;
- **J2b** (undo the void winner's back-fill, scan the anchors it skipped): in 80 — at the same candle 6, EARLIER 7
  (a later anchor's breakout applies before the void winner did — today the void winner pre-empts it), later median 1
  (… 33); never in 32.

Example — WAVES 302 / 167 (synthetic H1, sd +1):

| candle | high | low | close | today | J2b |
|---|---|---|---|---|---|
| 26 | 0.62494 | 0.62064 | 0.62482 | cycle 1 ESTABLISHED (CTS 26) | same |
| 27 | 0.62831 | 0.6247 | 0.62812 | a `one_maru_continuous` anchored here — no range → threshold None | same |
| 28 | 0.62883 | 0.62608 | 0.62847 | raw CTS → 28 (0.62883) | same |
| 30 | 0.62784 | 0.61289 | 0.62733 | cycle 1 CONFIRMED by proximity (anchor 28); `cts_threshold` 0.62883 | same |
| 32 | 0.63016 | 0.62613 | **0.62951** | the 27-breakout applies: cycle 2 ESTABLISHED (CTS 32, 0.63016; BOS 30) | VOID (candle 27 closed 0.62812, not above 0.62883); scan from 28: a `one_maru_continuous` anchored at **32** (close 0.62951 above 0.62883) |
| 36 | 0.63453 | 0.62914 | 0.63446 | — | it applies: cycle 2 ESTABLISHED (CTS 36, 0.63453; BOS 30) |

---

## 4. Implementation notes (for the build, after round 1)

- J1's restore is a snapshot before a range back-fill and a restore after a non-terminal one — the frozen look-ahead
  then serves only the terminal-reversal check (and, under J1c, the seed). Not a rewind of the run: the candles are
  re-stepped today already; only the state they read changes. The snapshot must keep the embedded FibTracker's event
  identities (the order tripwire compares by identity) and restore IN PLACE (`_post_apply_range_check` holds a local
  alias of `self.state`).
- J2 re-runs `detect_best_for_anchor` once per mismatched establishment (0 on the window).
- J3 changes `_select_bos_on_breakout`'s signature (the CTS anchor); a window whose end precedes its start keeps
  today's swap and is logged by the variant.

---

## 5. Open questions — round 1 (asked 2026-10-08)

J3 needs no question — R9-2 (ii) decided it ("the cycle-T BOS window ends at the CTS anchor") and J3 is exactly that
(byte-identical on the window; F17 (ii) skips W 3 → 0, WB 10 → 6, the 6 left the F17 (i) shape J2 removes).

- **J1-Q1 — how far does "the state as of that candle" reach in a re-stepped candle?** (§1 J1, §2, §3.) Candidates:
  (a) **J1a — everything as of the candle, the range's bounds included**: the range confirmed at `c` is still active
  from its candidate candle `i` (the documented look-ahead), but it starts with the CTS as of `i` and each re-stepped
  candle widens it. Window: 4 CSVs (every range a back-fill moved the CTS in starts narrower — `RANGE_STARTED`'s meta
  CTS + extra `RANGE_UPDATED`s; one sub range BREAK at M15 2026-01-13 16:15; H1 row 310's CTS / range top), the H1
  range band at 2025-12-03 20:00; KL / POI / fib / WVMI unchanged. Corpus: 0 errors, every F16 trip gone (41 → 0);
  structure differs in 1.0 % (WAVES) / 1.6 % (REAL) of the subset. The F16 slice's threshold is 0.5723 at 13:30 and
  0.57191 only at 13:45. (b) **J1c — as of the candle, except the range's seed** = the CTS the look-ahead reached
  (today's seed): window = the event row order + 6 per-row debug cells only; corpus structure 0.4 % / 0.08 %
  (confirmations only), F16 gone too; but a range bound / `CTS_THRESHOLD_UPDATED` can carry a later candle's price —
  the F16 slice emits 0.57191 at 13:30 (candle 24's low, printed at 13:45). (c) **J1b — the CTS only** (no raw CTS
  update inside a range back-fill): == J1a on the window, but a proximity confirmation (WAVES 301 / 11) or a watch
  (the probe fixture) inside a back-fill still reaches the earlier candles. (d) **also remove the range-activation
  look-ahead** (a range exists only from its confirmation candle) — changes the Week-5 range rule; its own plan, not
  R9-1. **Recommended: (a)** — R9-1's rule without an exception; (b) keeps a look-ahead in the threshold.
- **J2-Q1 — what "breaks T−1's `cts_threshold` at the apply" means, and what follows a failed break.** (§1 J2, §3.)
  Candidates: (a) **J2b — the pattern must qualify against `cts_threshold` as it stands at the apply** (the detector's
  own test: its first candle closes beyond, plus the body / tail checks — the R3-1 mirror of the reversal's frozen
  barrier); a pattern that does not is VOID, its back-fill is undone and the anchors it skipped are scanned. 112
  voids / 72,000 corpus trials (0 on the window); after the void, cycle T comes at the same candle in 6, earlier in 7,
  later (median +1) in 67, never in 32. (b) **J2 — the same test, void → continue at apply + 1** (the skipped anchors
  are never scanned): cycle T median 6 candles later; never in 47 / 112. (c) **the apply candle's close beyond
  `cts_threshold`** — voids 8 / 112; the other 104 establish as today (WAVES 302 / 167: cycle 2 at 32, from the
  pattern anchored at 27 whose first candle closed below the threshold). (d) (c) + the J2b scan for those 8.
  **Recommended: (a)** — the same test every other establishment passes (19 / 19 on the window detect against the
  very threshold they establish on); the scan keeps the breakouts the void winner had pre-empted.

## 6. Stages (proposed — confirm with round 1)

One cause per stage, each its own local commit + `/compare` vs the previous stage's outputs + pins; the window
predictions are the variant replays (§2):
1. **J3** — the BOS window ends at the CTS anchor (byte-identical: 27 / 27, 3 / 3). Tests: the two direct callers of
   `_select_bos_on_breakout` (`test_ms_bos_selection_guards.py`) take the CTS anchor; a pin on WAVES 301 / 572's candles.
2. **J2** (J2-Q1's choice) — byte-identical on the window (0 mismatched establishments). Brings the snapshot / restore
   helper (J2b) that J1 reuses. Tests: `test_plan_i_s2b_dispatch_order::test_the_f14_1_shape_from_a_real_ms_run_now_
   stops_the_downstream` re-derived (MS now voids that shape); pins on 413 / 1074 and 302 / 167.
3. **J1** (J1-Q1's choice) — the window cells of §2 for that variant; the H1 chart change rendered for review first.
   Tests (census, §2): the F16 raise pin retired (→ a pin that the real slice confirms CTS 21 at 13:30), 8 re-derived
   (event order, the probe fixture's early stop — Plan B's quiescence now reads the state as of the candle).
Then the docs (MARKET_STRUCTURE_SPEC "Look-ahead windows without rewinds" + "BOS_n+1 derivation rule" + the breakout
establishment rule; LANDMINES; CROSS_CYCLE_FIB_SPEC §1 / §8 / §9 — the order tripwires stay as guards; PLAN_I F16 /
F17 → fixed) and the random-tail corpus re-run on the built code.

## 7. Decisions

**2026-10-09 (round 1, §5)**

| ID | Decision | The user's words |
|---|---|---|
| J1-Q1 | (a) **J1a**: a re-stepped candle reads EVERYTHING as of itself — the CTS, a confirmation, the BOS barrier / a watch, and the range's bounds (the range confirmed at `c` stays active from its candidate `i` — the documented activation look-ahead — but starts with the CTS as of `i` and is widened by each re-stepped candle); a terminal back-fill (a reversal inside it) unchanged | "(a) J1a: all as-of (Recommended)" |
| J2-Q1 | (a) **J2b**: a breakout establishing cycle T ≥ 1 whose detection threshold differs from `cts_threshold` at its apply must pass the detector's own test against that threshold at its anchor (same pattern, same apply); else VOID — its back-fill undone, the anchor candle stepped, the anchors it skipped scanned | "(a) J2b: re-test + rescan (Recommended)" |

J3 was decided by PLAN_I R9-2 (ii). The stage order of §6 (J3 → J2 → J1) was proposed with the round.

## 8. Change log

- 2026-10-08 — opened (session 6, after PLAN_I S5b `515c449`): the three causes, the window census, variants J1a /
  J1b / J1c / J2 / J3 on the window (J2 / J3 byte-identical; J1 NOT — 4 CSVs + the H1 range band under J1a; J1c
  event order + 6 debug cells), the concrete examples.
- 2026-10-08 → 09 — the corpus (base reproduces session 4; J1a / J2 / J3 / J1a + J2 + J3 — the combination 0 errors,
  0 trips, 0 skips over 72,000 trials), the structural signature subset, J2's 112 voids (104 with the apply candle
  closing beyond), J2b (the rescan); the suite census per tree. Round 1 asked and ANSWERED (§7): J1a, J2b.

## 9. Build record

**J3 — the cycle >= 1 BOS window ends at the new CTS anchor (PLAN_I R9-2 (ii), F17 (ii)) — LANDED** (its own local
commit).
- Code: `MarketStructure._select_bos_on_breakout(breakout_apply_idx, cts_anchor_idx)` — the window `[pullback apply |
  proximity confirmation, cts_anchor_idx]` (was `…, breakout_apply_idx]`); the swap of a window whose end precedes its
  start kept (the F17 (i) shape — J2's); the call site passes the breakout's CTS anchor; the emitter's comment.
- `/compare` vs S5b (= `%TEMP%/pi/S5`): **byte-identical** — 27 / 27 CSVs md5, 3 / 3 figures (full JSON); run.log
  9,101 lines == but the timings; fetch gate PASS. = the prediction (the J3 variant: 0 window-order swaps; the
  session-4 corpus: F17 (ii) skips W 3 → 0, WB 10 → 6 — the 6 the (i) shape).
- Tests 1427 → 1431: `test_plan_j_j3_bos_window.py` (2 — the real MS run on WAVES 301 / 572's candles: cycle 1's BOS
  39 (0.59943) ≤ its CTS 43, the window's start 7; the downstream dispatch takes it — both fail on the base tree);
  `test_ms_bos_selection_guards.py` — the two direct callers pass the CTS anchor, + the window ends at the CTS anchor
  (candle 10 made the deepest: excluded with the CTS anchor at 9), + the swap (CTS anchor 5 before the start 7 →
  `[5, 7]`).
- Docs: MARKET_STRUCTURE_SPEC "BOS_n+1 derivation rule" (+ the tripwire paragraph), GLOSSARY ("proximity-confirmed
  CTS", FibEventOrderError), LANDMINES (the proximity-cascade signature, the dispatch entry, the tripwire rule),
  ARCHITECTURE (the `BOS_CONFIRMED` row), CROSS_CYCLE_FIB_SPEC (§1, §8, §9), PLAN_I §9.2 F17.
- Mutation (solo, `mutant_runner.py`, full suite per mutant): 4 + a no-op control — all 4 KILLED (the window ends at
  the apply again; the swap removed — also caught by S2b's F14 (1) real run, which goes through the swap; the call
  site passes the apply; the end at the CTS anchor − 1), the control SURVIVED.
