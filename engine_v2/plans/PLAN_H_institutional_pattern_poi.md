# Plan H — One POI per cycle: the institutional-pattern IC (Week 9 Part 1 item 1)

**Status: SPEC DECIDED (2026-10-05, after 6 rounds); the implementation plan (§8) REVIEWED and folded (2026-10-06, §8.8 + the
user's R1–R4) — two landings (L1); landing A BUILT 2026-10-06 (§8.3a), its `/compare` == the prediction, committed
`e1c597c` + saved `20261006_120513_e1c597c` (S_A, R4); §8.4 measured (§8.4a); landing B BUILT 2026-10-06 (§8.5a), its
`/compare` == the prediction, committed `e54f509` + saved `20261006_151106_e54f509` (S_B); chart-reviewed → the
chart-debugging session (§8.9: F1 the M15 fib layer + F1·1) PAUSED 2026-10-06 for the cross-cycle fib revamp
(register §D.1 item 2; §6 F2a–F2e) and saved `20261006_224610_42c137c` (S_F1, the revamp's baseline); B's landing
review RAN at the revamp session's start (2026-10-06c, §8.5c: no BLOCKER / MAJOR; the fix-up commit byte-identical
to S_F1); the debugging resumes after the revamp — the revamp COMPLETE 2026-10-09 (PLAN_I S1–S6 + PLAN_J): re-derive
the logged points first (PLAN_I §9.9), each experiment vs **S_S6** `20261009_020650_5f6bae3` (the new baseline; S_B / S_A
stay named references — their fib CSVs carry the pre-S6 `fib_mode` column, the `/compare` skill's fib row).** Living document: every question, answer and decision
of the item-1 discussion is recorded here as it happens (§5 open questions, §6 decisions). Register line:
`IDEA_PARKING_LOT.md` §D.1 item 1.

**Input (the user's spec):** `plan_h_inputs/institutional_pattern_spec.md` + figures
`plan_h_inputs/institutional_pattern_strong.png` (single-candle bases) and `institutional_pattern_weak.png` (a 2-candle
middle segment). The repo copy is CANONICAL (H3) and is amended in place as decisions land (each amendment cites its
decision ID); the original `artifacts/institutional_pattern_prompt.md` is gitignored and no longer read. The spec holds
the user's rules in prose; §1 below is the precise engineering reading. Both follow §6.

**Prototype:** `plan_h_inputs/ip_proto.py [H1|confluence|counter] [cut|run]` (round 6: the H1 save frame, or an M15
lens frame rebuilt from the save's M15 chart; every decided reading, D3c as a switch — `cut` = the decided reading;
run from the repo root with `PYTHONPATH=.`); `ip_proto_round5.py` = round 5 (switches `[run|last] [include|best|trim]`);
`ip_proto_round4.py` = round 4 (`[agreed|off] [best|ab] [skip|fail]`); `ip_proto_round3.py` = the round-3 readings;
`ip_proto_round1.py` / `ip_asof_round1.py` = the round-1 readings (superseded, kept for the record of §3's numbers).

**Principle (user, 2026-10-02, D4):** every IC condition (§3.1, the zone touch, and later §3.2 if ever tested) stays
OUTSIDE the institutional pattern and is applied first; the pattern search runs LAST, around the surviving candidates.

---

## 1. The engineering reading (corrected as answers land)

**Goal.** Replace the V30 / V60 / V90 multi-IC selection (up to 3 ICs per fib) with ONE IC → ONE POI per cycle, chosen
by an *institutional pattern* (IP). All other IC conditions stay (POI_ZONES_SPEC §3.1). Every TF, main + sub.

Per fib (BOS anchor `b`, CTS anchor `c`, direction `sd`):

1. **Candidates** = §3.1 (`b ≤ i ≤ c`; **agreed direction** −sd (VD — today `direction == −sd`); an unfilled sd
   imbalance in `(i, c]`, fill horizon `c`, `evaluated_at=None`) **+ touches the 61.8–80 zone**, inclusive
   (`h ≥ zone_lo and l ≤ zone_hi`). §3.2 stays unapplied (D4).
2. **Leg candles: the agreed direction (V1 / V1a).** Every candle gets ONE direction for the legs from its candle
   `direction` (`sign(c − o)`) and its `vol_dir` (`sign(c − prev c)`, `features/candles_v2.compute_volume_features`):

   | `direction` | `vol_dir` | agreed direction |
   |---|---|---|
   | +1 / −1 | the same, or 0 | the candle's direction |
   | +1 / −1 | the opposite | none — a **split candle**: belongs to no leg and breaks every run |
   | 0 | +1 / −1 | the volume direction — a **volume-only doji** is a directional candle |
   | 0 | 0 | 0 — a **true doji** (item 5) |

   ("When one of them is 0, we can simply take the direction of the other.") Initial / continuation legs = agreed sd;
   opp leg = agreed −sd; single and multi legs alike. The IC: agreed −sd (VD, 2026-10-05) — §3.1's direction
   condition reads the agreed direction, so a volume-only doji can be the IC (type `pinbar`: body 0) and a split candle
   cannot. Rows 1–2 = GLOSSARY "qualified candle" (wave candles, WVMI — they keep it for now, Q-W: "we likely will
   also make the change"); row 3 is new. Names decided (N5). Implementation: ONE helper (the "qualified" rule is
   already coded twice — `wave_candles._is_qualified` and inline in `wvmi._find_temporary_lp`; the IP must not add a
   third copy).
3. **The IC is the LAST candle of its opp leg**: after the IC (and any true dojis), the next candle must be an agreed-sd
   candle; another opp candle (a volume-only opp doji included) means the candidate is not an IC, any other candle
   means no pattern (VD: such a doji is itself the next candidate).
4. **Legs** (P4 / P5 / B1 / B2 / B1a):
   - opp leg = the whole run of opp candles ending at the IC (no cap);
   - initial leg = the sd run immediately before the opp leg: its most recent `min(run, cap)` candles (cap 1 for
     3D, D3c: item 9); a run reaching before `b` is CUT at `b`;
   - continuation leg = the sd run immediately after the IC: its first `min(run, cap)` candles; CUT at `c`;
   - cap (per-TF table, H1): **H1 3, M15 3** (M5 2 later);
   - a leg is never shortened to improve the rank: the runs are fixed per IC; the only alternatives are the
     placements of true dojis at the initial↔opp junction (item 5, J1) and the 3D cap-1 initial next to the cap-3
     one (D3c) — the best wins, so each IC still yields one pattern or none. Outer-edge true dojis are trimmed (E1).
5. **True dojis (direction 0 AND `vol_dir` 0; P6 / P6a–P6e / P6c-final / J1):** never break a run, and are FULL
   members of the leg they are ASSIGNED to — they count toward the cap and make that leg multi (P6b / P6c: IC · ⊖ ↑ ↑ ↑
   → the continuation is ⊖ ↑ ↑, multi). "We simply need to assign them to the right legs." Inside a leg → part of
   that leg. Between the
   opp and continuation legs → always the continuation ("we do not want a doji as the IC, so never as the last candle
   of the opp leg"; ↑ ↑ ↑ ↓ ⊖ ↑ ↑ → a single opp leg and a multi continuation). **Between the initial and opp legs (J1 =
   a, 2026-10-05): tried in EACH leg; the better (rank, shape) wins**, each placement following that leg's cap rules —
   joining an initial leg at the cap pushes its earliest candle out (the cap window slides, so `pattern_anchor_idx` and
   the big reference move with it), and a doji can never join a single-only (3D) initial leg; a full tie (same rank and
   shape) → rule A / B (the initial leg if below the cap, else opp). Several junction dojis: every split is tried.
   **Never at the pattern's outer edge (E1, 2026-10-05):** a true doji is never the initial leg's first candle nor the
   continuation leg's last — "even if say those legs are already multi and their caps have not been met" — so it is
   trimmed off; also where the cap cuts a run on it (supersedes P6d's "a pattern may start on a doji"; reading →
   E1-confirm): ↑ ↑ ⊖ ↑ ↑ · opp → the initial leg ↑ ↑; IC · ↓ ↓ ⊖ | ↓ (sell) → the continuation ↓ ↓ (§3 round 6).
6. **Stacking** a leg: O = first open, C = last close, H = max high, L = min low. A 1-candle leg = the candle itself
   (its `candle_type`). A stacked candle is classified by the SAME classifier and parameters as a single candle (P2 /
   item 5): `classify_candles` with the per-TF `CandleParams` — body-% bands, special-maru promotion, per-TF pip floor.
7. **Big flags** of any leg (item 5 / P3): the single-candle definition and parameters — ratio =
   `round(stacked length / ref, 2)` with ref = the max length of the `lookback` (5) marus strictly before
   `pattern_anchor_idx` (the initial leg's first candle; = `prior_maru_max_len_as0` there, the breakout patterns'
   convention); big_maru = type **maru** AND ratio ≥ `big_maru_threshold` (0.65); big_normal = type **normal** AND
   ratio ≥ `big_normal_threshold` (0.5) (N2). Implementation reuses the candle-feature functions — no re-typed thresholds.
8. **Pattern requirements:** continuation leg big_maru AND its stacked body > the opp leg's stacked body (strict, P10).
9. **Ranks** (letter, then shape; R1):

   | Rank | Initial | Opp (IC) | Continuation | Shapes (initial-opp-continuation) |
   |---|---|---|---|---|
   | 1A | big_maru | maru | big_maru | i s-s-s, ii m-s-s, iii s-s-m, iv m-s-m |
   | 1B | maru | maru | big_maru | same |
   | 2A | maru | normal | big_maru | same |
   | 2B | big_normal | maru | big_maru | same |
   | 2C | big_normal | normal | big_maru | same |
   | 3A | maru | pinbar | big_maru | same |
   | 3B | big_normal | pinbar | big_maru | same |
   | 3C | maru | multi (≥ 2) | big_maru | i s-m-s, ii m-m-s, iii s-m-m, iv m-m-m |
   | 3D | big_normal, **single only** | multi (≥ 2) | big_maru | i s-m-s, ii s-m-m |

   A single opp leg's type = the IC's `candle_type`; a multi opp leg has no type requirement (P8a: "we are always taking
   1 candle of the stack as the IC, which is always weaker than if the IC was 1 true single candle"). 3D ranks last.
   **The 3D initial leg (D3c, 2026-10-05): cap 1** — the single big_normal candle immediately before the opp leg
   (after the junction placement), whatever sd candles precede it, never a stack; it is evaluated NEXT TO the cap-3
   initial leg (the same run's most recent ≤ 3) that the other ranks use, and the better pattern wins ("it doesn't mean
   we cannot use the big_normal candle as a part of a stack for a multi initial leg if we can achieve a higher ranked
   pattern"). Cap 1 is ONLY for 3D's initial leg ("Cap 1 is for 3D only and is ONLY for 3D's initial leg"): 2B / 2C /
   3B keep the cap-3 (multi-allowed) initial, and an opp RUN — consecutive opposite-direction candles — is never cut
   down to its IC to read as single (P5: ↑ maru · ↓ · ↓(IC) · ↑ big_maru is 3C, never 1B on the IC alone). A true doji
   at the junction is not an opp candle: it is placed (J1) — big_normal · ⊖ · ↓(IC) · ↑ big_maru: ⊖ → initial may
   stack to a maru and keep the opp leg single (1B / 2A / 3A), ⊖ → opp gives 3D.i; the better wins. Unlisted
   combinations → no pattern (P8).
10. **Tiebreakers among patterns of the same rank + shape — UPDATED 2026-10-09 (§6 TBU-1 … TBU-5, §8.9 "The IP
    tiebreaker update"): a NEW first one, the immediate unfilled imbalance (an unfilled sd imbalance whose first gap
    candle is at IC + 1 / IC + 2 — a filter: one passing → final, none → all go on), then the two below, renamed
    `ic_candle_type` (TB1) and `most_recent_before_imbalance` (TB2); canonical POI_ZONES_SPEC §3.4.** As decided
    in round 4 (TB, 2026-10-02 — supersedes the round-3 list and R4 / R6): TB1 IC type maru > normal > pinbar (the IC candle itself; only separates 3C / 3D) → TB2 the most recent
    IC before the OLDEST unfilled sd imbalance. TB2 always ends it (distinct idx). No tie params (the 80 % bands are
    gone). **TB2a (decided 2026-10-05):** the reference = the earliest unfilled sd imbalance AFTER the oldest tied IC
    (its first c2 > that IC; one always exists — every candidate has one); the pick = the most recent tied IC before
    the reference's first c2 (it may be the reference's c1, also for a multi-candle gap — possible only because that
    candle is −sd). "Unfilled" = §3.1's own test (fill 0.70, horizon = the CTS anchor, and the same `evaluated_at` as
    the caller's §3.1 question). User example: tied 4083 / 4136, unfilled 4029, 4132, 4151 … → reference 4132 → 4083;
    without 4029 / 4132 → reference 4151 → 4136.
11. **No pattern → no POI** (D7): MS + the proximity scan see only the BOS inner for that cycle.
12. **POI = the IC candle's high / low** (P9; item 3 revisits size).
13. **One selector, two consumers (D2):** `derive_poi_zones` (drawn; the final fib) and the MS in-flight resolver
    `compute_poi_inners_for_cycle` (sd-proximity CTS confirmation; the as-of fib) call the SAME selection function and
    read ONE parameter source (today the orchestrator re-types the `POIConfig` literals while the resolver uses the
    defaults — same values, two sources; the new params must have one).
14. **Activation (D1 = today's shape):** pick once on the final fib; the activation sweep decides when it is live. Its
    per-CTS-event re-check becomes "the IC touches the as-of zone" (reason `ic_not_touching_fib_zone`, D1a / R3), and
    `first_active = max(CTS-established moment, ic_idx, lifecycle floor, pattern_confirmed_idx)` (D1b / R1 — the moment
    the continuation leg is known complete: its last candle when it hit the cap, else the next candle).
15. **Who reads the POI (the cascade paths, 2026-10-05 — from the code):** (a) MS sd-proximity CTS confirmation via
    the in-flight resolver (main + subs); (b) the zone-proximity scan `check_zone_proximity` — its sd trigger reads the
    NEAREST active POI inner (buy: the highest; sell: the lowest; else the BOS inner), and those triggers are the H1
    WVMI gate (the first sd trigger per cycle) AND the multi-TF `first_counter` / `subsequent_*` triggers that start the
    M15 subs; (c) the charts' POI hover and `*_poi_zones.csv` (`versions`). So a cycle whose nearest POI today is a pick
    the IP does not choose moves its sd triggers (§3 round 5).

## 2. What changes vs today

| | Today | Decided |
|---|---|---|
| Candidates | §3.1 base conditions (§3.2 never implemented) | §3.1 with the agreed direction + touches 61.8–80 (+ the IP legs, applied last) |
| Selection | per threshold V30 / V60 / V90: the MOST RECENT candidate meeting it → 1–3 ICs | best IP (rank → shape → TB1–TB2) → 0–1 IC |
| POIs per cycle | 1–3 (one per unique picked IC) | 0–1 |
| Shared with MS | `select_ic_variants` feeds `derive_poi_zones` AND the MS in-flight resolver | the IP selector feeds both (D2) |
| Meta | `versions` / `current_versions`; reason `variant_below_v30` | `ip_rank`, leg candle counts, leg idx, `pattern_confirmed_idx`; reason `ic_not_touching_fib_zone` (rule-3 atomic migration, D6; R1–R3) |

## 3. Measured (save `20260930_151531_8b11bc2`)

**Round 6 (2026-10-05; E1 = true dojis trimmed off the pattern's outer edges incl. a cap cut, D3c = the 3D cap-1
initial) — picks UNCHANGED on all three lenses** (D3c cut and run; diffed against round 4).

- **E1 trims one leg on the window** — a cap cut, the case P6d used to allow. Confluence sub 4 cyc 0 (sell),
  candidate 3463 (↑ maru): continuation run 3464 ↓, 3465 ↓, **3466 true doji** (o = c = prev c 0.57686), 3467 ↓ → the
  cap of 3 ends the leg on the doji (3464–3466), E1 trims it → 3464–3465. The candidate fails anyway (initial 3462 is a
  pinbar). No outer-edge dojis elsewhere.
- **D3c:** the cap-1 initial gives a valid 3D on the same 4 candidate rows as round 5 (conf 2415 / 4072, counter 3654 /
  4072), each already a 3C → no candidate's best changes.
- **Why cap 1 must stay 3D-only (a probe, not a reading):** letting EVERY rank also try the run's last candle alone
  would improve 16 candidate rows by shape (multi → single initial; H1 2, confluence 9, counter 5), e.g. H1 832 3C.ii → 3C.i, conf 2442 2A.ii → 2A.i,
  4083 1A.iv → 1A.iii — the P4 rule ("max consecutive available, up to 3") is what keeps those multi.

**Round 5 (2026-10-05; VD = a volume-only doji can be the IC, J1 = best placement with the sliding cap, + the open D3c
/ E1 as switches) — picks UNCHANGED on all three lenses under every switch** (D3c run / last × E1 include / best / trim;
diffed against the round-4 output).

- **Candidates (VD):** the agreed direction ADDS volume-only dojis (candidate rows: confluence 15, counter 8, H1 0 — all
  type `pinbar`, body 0) and DROPS split candles (confluence 4, counter 3, H1 0). None of the added ones forms a pattern
  (most are followed by another opp candle; the rest fail the continuation); none of the dropped ones had one.
- **The VD example (conf sub 1 cyc 4, sell):** 2316 (o = c 0.57809, h 0.57816, l 0.57780) is now the candidate instead
  of 2315 — but its low misses the zone top 0.57779 by 0.1 pip, so it is not a candidate, and 2315 is not the last opp
  candle → this leg gives no pattern (round 4: a pattern failing its continuation, 8.7 vs 8.9 pips). The pick stays
  2320 2A.iv.
- **J1:** 0 junction dojis next to a candidate on any lens (as in round 4). **E1:** 0 outer-edge dojis on any
  candidate's legs.
- **D3c:** the cap-1 reading gives a valid 3D on 4 candidate rows (conf 2415 / 4072, counter 3654 / 4072), each already
  a 3C through its 3-candle initial (3C ranks above 3D) → it changes no candidate's best here. Example counter sub 5
  cyc 0 (buy): up run 3648–3651 (4 candles; the leg = 3649–3651, stacked maru) · opp 3652–3654 · continuation
  3655–3657 → 3C.iv; the cap-1 reading would also accept 3651 alone (o 0.57326, c 0.57336: body 1.0 p, length 2.4 p,
  a big_normal) as a 3D.ii initial.
- **The cascade through the proximity scan (§1 item 15) — corrects round 3's "sid 0 cyc 1: MS unchanged", true for MS
  but not for the triggers:**
  - H1 sid 0 cyc 1 (buy): the first sd trigger today is **710** (low 0.57902 ≤ POI 611's inner 0.57850 + 8 p). It is
    the H1 WVMI gate (`_wvmi.csv` `triggered_by_event_idx` 710) AND counter sub 3's `first_counter` trigger (M15 2843 =
    the 12:45 candle of H1 710). With 607 (inner 0.57712) the first candle after the `CTS_CONFIRMED` 683 with low ≤
    0.57792 is **748** (low 0.57764, 2025-12-31 02:00; the scan runs to the sid 0 reversal, ~897) → both move ~38
    hours later (M15 2843 → 2995).
  - H1 sid 1 cyc 2 (sell): the 926 sd trigger is also the WVMI gate and counter sub 5's `first_counter` (M15 3707) →
    **939** (M15 3759), with the CTS confirmation move already recorded (round 3).
  - Every later sd / opp_sd trigger of those two cycles can move with them (confluence sub 6 @3819 and sub 4's end,
    counter sub 7 @4083, confluence sub 7 @4200) → **6 of the 11 trigger records** (all 3 counter, confluence 4 / 6 /
    7) are re-derived; the round-4 M15 table is firm only for confluence subs 0–3. The exact trigger lists: a shadow run
    of `check_zone_proximity` with the IP picks, when the plan is written (predicted cells).

**Round 4 (2026-10-02; the agreed direction, dojis as full members, the imbalance tiebreaker) — H1 + both M15 lenses.**

- **H1 picks unchanged: 607 / 732 / 832** (table below), under every switch.
- **M15 frames** are not in the save CSVs: rebuilt from the save's M15 chart (the `OHLC` trace + the two
  imbalance-candle traces = 4,228 contiguous candles; the OHLC trace alone has holes), features recomputed with the
  M15 1.0-pip floor. Index check: all 42 saved M15 POI ICs' high / low match the frame at the same idx. Fibs: each
  lens's `*_fib_lifecycle.csv` (active or locked), the drawn fib; lifecycle bounds NOT applied (caveat below).
- **Picks are identical under every switch on all three lenses** (agreed vs candle-direction-only; junction best vs
  rule A / B; VD skip vs fail): on this window the volume rule, J1 and VD change no pick.
- **Candle counts.** H1: direction 0 = 13 (5 true dojis + 8 volume-only dojis), split 20. M15: direction 0 = 100 (41
  true + 59 volume-only), split 50 (23 of them the 22:00 rollover candle, median body 0.2 pip), directional with
  `vol_dir` 0 = 62. Next to a candidate: true doji at a junction 0; volume-only doji right after a candidate 9
  (confluence) / 5 (counter), every one failing for another reason.
- **M15 today vs the pick** (unique sub cycles; "both" = the same fib in both lenses):

  | M15 cycle (drawn fib) | Today's ICs | Pick |
  |---|---|---|
  | conf sub 0 cyc 0 / 1 / 2 (454 → 784 / 1020 / 1245) | 678 | none ×3 |
  | conf sub 0 cyc 3 / 4 (454 → 1721 / 1794) | 736 | 687 3C.iv (both) |
  | conf sub 1 cyc 0 (1797 → 1816) | 1810 V30, 1808 V60/V90 | none |
  | conf sub 1 cyc 1 (1837 → 1898) | 1890 V30/V60, 1889 V90 | **1875** 3C.iv (TB: 1842 / 1875) |
  | conf sub 1 cyc 2 (1934 → 1945) | — | none |
  | conf sub 1 cyc 3 (2078 → 2231) | 2181 V30/V60, 2106 V90 | 2106 1B.iii |
  | conf sub 1 cyc 4 (2270 → 2365) | 2320 | 2320 2A.iv |
  | conf sub 2 cyc 0 (2365 → 2557) | 2433 | 2383 3A.iv |
  | conf sub 2 cyc 1 (2365 → 2609) | 2445 V30, 2433 V60/V90 | 2442 2A.ii |
  | both sub 3 cyc 0 (2639 → 2736) | 2706 V30/V60, 2702 V90 | 2704 1A.i |
  | both sub 3 cyc 1 (2758 → 2842) | 2808 | 2798 2B.iii |
  | both sub 3 cyc 2 (2758 → 3047) | 2961 | 2837 1A.iv |
  | both sub 3 cyc 3 (3304 → 3611) | — (fib live 3589–3611 only) | (3388 1A.iii — lifecycle not applied) |
  | conf sub 4 cyc 0 (3304 → 3621) | 3463 V30/V60, 3458 V90 | 3388 1A.iii |
  | counter sub 5 cyc 0 (3621 → 3760) | 3654 | 3654 3C.iv |
  | conf sub 6 cyc 0 (3760 → 4000) | 3809 V30/V60, 3808 V90 | none |
  | both sub 7 cyc 0 (4027 → 4086) | 4048 | 4035 2C.iii |
  | both sub 7 cyc 1 (4027 → 4157) | 4131 V30, 4127 V60, 4116 V90 | 4051 2A.iv |
  | both sub 7 cyc 2 (4027 → 4222) | 4145 V30, 4136 V60, 4127 V90 | **4083** 1A.iv (TB: 4083 / 4136) |

  Of the 20 cycles with a POI today: the same IC in 3, a different IC in 12, no POI in 5. M15 picks are mostly maru /
  normal ICs (1A–2C) — unlike H1's pinbar ICs (the 1.0 vs 2.2-pip floor).
- **TB exercised twice.** conf sub 1 cyc 1: 1842 and 1875 both 3C.iv; the oldest unfilled bear imbalance after 1842 is
  c2 1892–1893 (1873 / 1877 / 1882 filled by 1898) → 1875 (the round-3 rule gave 1875 too). Sub 7 cyc 2 (buy): 4083
  (maru, len 4.1p, body 2.9p) and 4136 (maru, len 10.9p, body 7.5p) both 1A.iv; unfilled bull imbalances as of 4222:
  c2 4029–4030, 4132, 4151, 4174–4176, 4195–4196, 4217, 4221–4222 → the oldest AFTER 4083 is 4132 → **4083**; the
  round-3 rule (length band) would give 4136. The FIB-WIDE oldest (4029–4030) precedes both ICs → TB2a.
- **The second POI-ridden confirmation (conf sub 1 cyc 1 `CTS_CONFIRMED` 1902, today IC 1889):** the pick 1875 has
  inner 0.58074 (1889: 0.58090); on the drawn fib the first candle within 5 pips after CTS 1898 is still **1902** →
  likely survives. MS's snapshot uses its own (intra-cycle) fib on subs (D5) → the replay confirms.
- **Touch edge (FYI, kept as decided):** an earlier opp candle touches the zone while the IC (the last) does not —
  2 cases with a valid pattern (2375 3C.iii in sub 2 cyc 0 / 1; 4048 3C.iv in sub 7 cyc 2); both outranked, no pick
  changes. The touch stays on the IC (the POI is the IC's range).
- **Caveat:** drawn fibs only, no lifecycle bounds (sub 3 cyc 3's fib is live 3589–3611 and has no POI today), no
  as-of (per CTS event) picks.

**Round 3 (2026-10-02; + the volume rule, the doji rules, the new tiebreakers): picks UNCHANGED from round 2 under
every volume reading (off / `vol_dir` 0 allowed / strict).**

| H1 cycle (fib) | Today's ICs (inner) | Patterns found | Pick (inner) |
|---|---|---|---|
| sid 0 cyc 1 buy (BOS 591 → CTS 652) | 611 V30 (0.57850), 607 V60/V90 (0.57712) | 593 3A.iv, 607 3A.iii, 611 3A.iv | **607** 3A.iii: 606 · 607 pinbar · 608–610 (0.57712) |
| sid 1 cyc 1 sell (689 → 761) | 732 (0.58068) | 732 3A.iv | 732 3A.iv: 729–731 · 732 · 733–735 — unchanged |
| sid 1 cyc 2 sell (826 → 905) | 865 V30/V60 (0.57720), 860 V90 (0.57813) | 832 3C.ii | **832** 3C.ii: 828–830 · 831–832 · 833 (0.57866) |

- **Volume rule (V1):** 20 of 1,045 directional H1 candles have `vol_dir` opposite to their candle direction, 17 of
  them the 22:00 UTC daily-rollover candle (the gap at the daily open); 10 have `vol_dir` 0. Example — 864
  (2026-01-07 22:00): opens 0.57748 after 863 closed 0.57728 (2-pip gap up), closes 0.57738: a DOWN candle with
  `vol_dir` +1. It breaks the sell run 862–864, so today's IC 865 now fails one step earlier (no initial leg) — it failed
  anyway (continuation 866 ratio 0.44). 865 itself (up 0.1-pip body, close == prev close) has `vol_dir` 0 — an IC only
  if V1a allows 0.
- **Tiebreakers not exercised on H1:** every cycle's best rank + shape is held by one IC. M15 unmeasured.
- **Dojis:** none of the 13 H1 dojis sits in or next to a pattern.
- **Every H1 pick is a pinbar IC** (3A / 3C): small-bodied candles are pinbars under the 2.2-pip floor (item 4
  recalibrates it → ranks will move then).
- **Why candidates drop out** (sid 1 cyc 2, 19 candidates → 1 pattern): most are not the last candle of their opp leg
  (e.g. 831 maru inside the zone is followed by −sd 832); the rest fail the continuation leg.
- **Round 1 → round 2:** sid 0 cyc 1 pick 605 → 607 because of P3 (continuation 606's ratio is 0.42 against the 5
  marus before 604, the pattern's first candle). Shapes moved toward multi under P4 (732 3A.ii → 3A.iv, 832 3C.i →
  3C.ii): H1 runs are often 3+ candles long.
- **Structure cascade (D2 accepted it):**
  - sid 1 cyc 2: today's `CTS_CONFIRMED` at 926 is sd-proximity on IC 865's inner 0.57720. With 832 (inner 0.57866) the
    first candle within 8 pips is **939** (h 0.57794 ≥ 0.57786; 906–935 max high 0.57750). So that confirmation moves
    to 939 at the earliest — or earlier via pullback — the replay decides. 5 of the 11 trigger records have this parent
    cycle (confluence 3611 / 3819 / 4200, counter 3707 / 4083) → re-derived.
  - sid 0 cyc 1: the nearest inner the proximity scan reads moves 0.57850 → 0.57712 (deeper); MS unchanged (that CTS
    confirmed by pullback at 683, and a deeper inner can only fire later) — but its sd TRIGGER moves 710 → 748 (the
    WVMI gate + counter sub 3's trigger; round 5).
  - The second live POI-ridden confirmation (M15 confluence sub 1 cyc 1 at 1902, IC 1889) — measured in round 4
    (above): likely survives.

**Round 1 (superseded readings, 2026-10-01 a.m.):** picks 605 2A.i / 732 3A.ii / 832 3C.i; re-picking at every CTS event
never switched IC (sid 1 cyc 1 has no pattern until the 756 update).

**Drawn POI vs MS snapshot today (D2 / D5, from the code — to be confirmed by a logged replay):** both run
`find_ic_candidates` + `select_ic_variants` with equal `POIConfig` values. At a cycle's last CTS event MS's snapshot
uses the drawn fib's anchors on main structures → the same ICs (except the accepted LANDMINES "Scenario 2 anchor
agreement" M1 drop case); at earlier CTS events it uses the then-current CTS anchor (by design, D1). On subs the fibs
differ on 8 unique sub cycles (D5) → different candidates. Example: confluence sub 0 cyc 2 (buy) — drawn fib BOS 454 →
CTS 1245, POI IC 678 (0.56234–0.56268); MS's fib BOS 1107 (0.57111) → CTS 1245 (0.57774), zone 0.57244–0.57364 —
~100 pips apart (that CTS confirmed by pullback at 1257, so no outcome diverged).

## 4. Naming (decided — N1–N3, N5)

- Legs: **initial leg / opp leg / continuation leg** — avoids "wave" (wave candles, WVMI), "pullback" (MS
  pullback-confirmed CTS, `pullback_th`, WVMI pullback momentum / LP / FP, the breakout patterns' pullback candle) and
  "base" (KL `base_pattern` / `base_idx`). The pattern: "institutional pattern (IP)".
- `pattern_anchor_idx` = the initial leg's first candle, `pattern_end_idx` = the continuation leg's last candle
  (GLOSSARY "Naming Standard": pattern-realm "anchor" = a pattern's FIRST candle).
- Rank label `ip_rank` = "2A.i" — "tier" already means chart opacity tiers.
- (R1 / R2, 2026-10-06) `pattern_confirmed_idx` = the moment the continuation leg is known complete (a Naming-Standard
  moment, `*_confirmed_idx`); `opp_leg_first_idx`; `ip_leg_candle_counts` ("a-b-c"); the tiebreak fields
  `ip_tb_runner_up_ic_idx`, `ip_tied_count`, `ip_tb_imbalance_first_c2_idx`; the cond5 reason `ic_not_touching_fib_zone` (R3).
- The leg direction (V1 / V1a, §1 item 2) extends GLOSSARY's **"qualified candle"** (rows 1–2) with the volume-only
  doji row, so it needs its own name (N5, decided 2026-10-05): **agreed direction**, **true doji**, **volume-only doji**,
  **split candle**.

## 5. Open questions — none (2026-10-05)

The round-6 items were answered (§6). Process notes for the plan (not questions): landing A changes nothing but adds
the debug CSVs (so no chart review after A — its check is the CSV vs the prototype; the hover lines HV go in with B);
landing B is the first engine change on this branch → its `/compare` names the Week 8 save explicitly (the TRAP);
GLOSSARY gains agreed direction / true doji / volume-only doji / split candle / institutional pattern / the three
legs / `ip_rank` (cross-linking "qualified candle") in the landing that introduces them; the `poi_zones.py` `versions`
meta comment ("peak variants") goes with landing B's migration (POI_ZONES_SPEC corrected 2026-10-05). Measurement
tool for later rounds: `plan_h_inputs/ip_proto.py` (rebuilds the M15 frames from a save's M15 charts).

**For the chart review (user, 2026-10-05; not questions now):** the cycles left without a POI ("we will need to come
back and understand if losing their POIs in these cycles is appropriate"); the touch-on-the-IC rule ("depending how
charts look I may ultimately adjust the rules, the pattern, and how we identify IC"); the counter lens re-derived
through the moved triggers (§3 round 5) — "We won't be able to judge whether those changes are appropriate or correct
until I can review the charts".

## 6. Decisions

**2026-10-09 (the IP tiebreaker update, before the user's holistic IC write-up — the spec verbatim
`plan_h_inputs/ip_tiebreaker_update_spec.md`; measured on `2d199fd` with `plan_h_inputs/tb_imb_measure.py` /
`tb_imb_dump.py`, §8.9 "The IP tiebreaker update")**

| ID | Decision | Why / the user's words |
|---|---|---|
| TBU-1 | A NEW first tiebreaker among the ICs tied on the best (rank, shape): **an unfilled sd imbalance immediately after the IC** — a filter: exactly one IC passes → it is final; two or more pass → only they go on; none → all go on. Then the IC's candle type (maru > normal > pinbar; was TB1), then the most recent IC before the earliest unfilled imbalance after the oldest IC still in play (was TB2). The type tiebreaker still runs between the two; in the user's example it changed nothing | "Prefer IC immediately followed by unfilled imbalance, "immediately" here means within 2 candles after the IC candidate (<= IC_idx + 2). This is essentially a filtering condition (true/false). All true IC candidates move onto tiebreaker 2 to be further narrowed down. If no trues, then all candidates move onto 2. If only 1 true, then that's the final IC & and we skip the rest." Why first: "a stronger & more stringent version of one of the 4 conditions … indicates immediate and large movement away from the IC … but ultimately, this by itself is not as strong of a signal as the institutional pattern, which is why it is applied as a tiebreaker after IP" |
| TBU-2 | "Within 2 candles" counts the imbalance's GAP candle — its c2; for a merged multi-candle imbalance its FIRST gap candle — at IC + 1 or IC + 2 (so the gap's c1 is the IC or the next candle), never the 3-candle pattern's c3. Consecutive sd gap candles merge, so at most ONE imbalance can sit in that window | "Q1 (a) is correct. It is always the gap itself instead of the 3 candle pattern this also applies for multi candle imbalances which would be the first gap candle" |
| TBU-3 | Descriptive names, not numbers, for the tiebreakers wherever they persist — code, the `ip_tiebreak` value, the CSV, the hover, the canonical spec; "TB1 / TB2 / TB3" stay planning shorthand (here and in the user's spec). The general rule: memory `feedback_naming_discussions.md` | "(ii) in general I want to move away from using nondescriptive naming with iterative numbers like TB2 or Scenario3 since when we come back to it far in the future, they don't provide any descriptive details. When I provide specs and discussion around logic, it helps to use these naming for organizational purposes … So their usage is strictly to help with understanding the ideas and plan our build". The names (agreed): `immediate_unfilled_imbalance` / `ic_candle_type` / `most_recent_before_imbalance`; the CSV column `immediate_imbalance_first_c2_idx`. No sweep of existing names; ordinal VALUES (the ranks 1A–3D, the shapes i–iv) stay: "there are places where its usage is appropriate like with the tier & ranks. The difference here is that these represent "values" of a variable. We should generally refrain from using them to "name" ideas, concepts, variables, etc." |
| TBU-4 | When two or more ICs pass the new tiebreaker (and the type tiebreaker leaves two or more), the last tiebreaker picks the OLDEST of them — exactly, by the pattern's structure: the IC is the last candle of its opp leg, so another tied IC can never sit before the oldest one's immediate gap | "TB3 oldest is intended" |
| TBU-5 | Accepted defaults: the tie = the same rank AND shape (e.g. both 1A.iv), as today; the debug CSV gets a per-candidate column = the first gap candle of its unfilled imbalance within IC + 2 (empty = none); when the new tiebreaker decides, the runner-up = the most recent tied IC that failed it, the imbalance column = the pick's gap, plus a hover line. Clarified + DECIDED: "unfilled" = condition 3's test, judged through the fib's CTS ANCHOR (the drawn POI: the final fib's; MS's read: its as-of fib's) — a gap before the anchor that fills only after it (e.g. 1662, filled at 1940 on the retrace into 1660's range) counts as UNFILLED, and a gap candle after the anchor is not in the window (the window = IC + 1 … min(IC + 2, the CTS anchor)); "2 candles" = RAW candle offsets, a true doji takes a slot (IC at t, a true doji at t+1, the gap candle at t+3 → fails) | "Everything else looks good to me." → "Ok sounds good on what you said on gap filled after the CTS does not count. true dojis included: yes raw counting is correct" |
| TBU-6 | The tiebreakers run ONLY among patterns tied on the best (rank, shape) — kept as built after the chart review: H1 sid 1 cyc 2 keeps 832 (3C.ii), its ONLY patterned IC candidate, although 832's immediate gap 833 (832 its c1) filled at 858 — no tie, so no tiebreaker runs. The user had expected 860: an IC candidate with an unfilled immediate gap (861–862, filled at 944, after the CTS anchor 905) but NO pattern — its continuation 861–863 is a maru of 9.3 pips with big ratio 0.63 < 0.65 (`continuation_not_big_maru`). Not adopted: the immediate imbalance as a fifth IC condition (832 cut → no other candidate with an immediate gap patterns → `IP: none`, no POI — not 860); revisiting the big_maru threshold | "Ok sounds good. Let's Keep it as built and we can commit." |

**2026-10-09 (RX1 round 1 — the IC reference conditions, §8.9 RX1; the spec `plan_h_inputs/ic_reference_conditions_spec.md`)**

| ID | Decision | Why / the user's words |
|---|---|---|
| RX1-B | The two requirements are ONE bundled condition (the price side + the before-the-break side), both keyed on the chosen reference price: no reference → the whole condition passes | "even though these originally were 2 separate conditions … I'd like to think of them more as bundled together as 1 condition / requirement. The reason is because they both depend on the reference price, so when there is no reference price, they both would pass" |
| RX1-1 | The bound for the previous structure's BOS (a reversal-started structure) = the valid, confirmed reversal pattern that caused the reversal (its anchor = the close-break candle: H1 897, sub 1 1936, sub 2 2465, sub 3 2824) — not cycle M's establishment pattern (wrong at M = 0: sub 2 2367) | "The pattern that broke the previous structure's BOS should be the valid / confirmed reversal pattern that caused the reversal" |
| RX1-2 | A structure started by a NON-reversal trigger that has a cross limit uses it as its previous-structure-BOS EQUIVALENT — the anchor behind `P` (sub 4: the parent's CTS_1 anchor 0.57400; subs 5 / 6 / 7: the previous sub's BOS anchor); every non-reversal trigger; no cross limit → no such price | "Yes anchor behind P is correct. Each time a structure starts without a reversal, IF cross limit exists, then the cross limit will effectively be used as the prev structure's BOS price. This should be applied to all non-reversal triggers. If a trigger does not have a cross limit, then there will be no "prev structure BOS"-equivalent price to reference" |
| RX1-2a | No break pattern reached (yet, or ever) → the before-the-break side passes ("every candle is considered before a break pattern") | "there may be cases where the price never breaks this reference price … If we have not reached a valid break pattern yet, then every candle is considered "before" a break pattern" |
| RX1-2b | The cross-limit level is not an MS key level, so MS has no break check on it: a NEW cross-limit BREAKOUT WATCH — the cross limit taken as a break threshold, a breakout pattern checked whenever price approaches / crosses it, like the reversal watch but on the BREAKOUT wave (the reversal watch is on the pullback wave). Reversals and new cycles already have their confirmed patterns in MS — reference those | "it's not a true MS key level, but just an arbitrary marker … So what I propose is that whenever we have a struct using cross limit as prev struct BOS-equivalent, we assume the cross limit as a break threshold, and we check for breakout pattern whenever price approach/crosses it. The idea would be similar to reversal watch when the price comes close to BOS threshold. However, in this case instead of the reversal watch happening on the pullback wave, the cross limit breakout watch is on the breakout wave." — details → round 2 |
| RX1-3 | Both references available → always the one further opposite the structure, even when the previous cycle's CTS already went past the BOS anchor price (inside the cross window — P further than the anchor, a BOS threshold expanded after its confirmation): sub 3 cyc 1 → 0.58198, sub 1 cyc 2 → 0.57882 | "the new conditions reference price should still be just the one further opposite the structure. In this case, that should be the BOS anchor price"; expected rare ("only … after BOS was confirmed, BOS threshold got expanded") — measured: 5 of the window's 8 limits are further than their anchor, 2 cycles in the gap |
| RX1-T | Terminology: "break" = a valid PATTERN crossing a threshold; a price that only crosses = "goes past" / "surpasses" (GLOSSARY "break vs go past") | "break implies not only the price surpassing a threshold, but a valid pattern that breaks a threshold. So here, when you refer to a cycle's CTS breaking the BOS anchor price, it has to mean just that the CTS went past the price" |
| RX1-W1 | The cross-limit breakout watch tests `P` (the cross-limit value — the moved threshold), not the anchor behind it: sub 7 → 0.57827 → the break is 4188 (4178 closes at 0.57820, short of it); sub 4 → 0.57376 → 3588. Same shape as a reversal, whose pattern breaks the threshold (sub 1: 0.57926) while the price side uses the anchor (0.57884) | first "I'm confused here what's the difference between these 2?" — explained with sub 7 (anchor 0.57806 = sub 6's BOS candle 3760; P 0.57827 = sub 6's threshold after the wick at 3806) → "P — the moved threshold (Recommended)" |
| RX1-W2 | A cross-limit breakout watch OPENS on a candle that CLOSES past its threshold in the structure's direction (like the reversal watch — never on an approach); the breakout pattern is anchored at that candle and must confirm within 5 candles | AskUserQuestion: "A close past the threshold (Recommended)" |
| RX1-W3 | A failed watch leaves the threshold FIXED; every later candle that closes past it opens a new watch until one confirms (sub 7 with P: 4183–4187 fail, 4188 confirms) — not the reversal watch's move-to-the-wick, not one watch only | "Threshold fixed, next close re-arms (Recommended)" |
| RX1-K1 | Two cross-limit breakout watches both confirming → the break = the EARLIEST CONFIRMATION (ties → the earlier anchor): the bound never moves once set (PLAN_K K-Q1) | AskUserQuestion: "Earliest confirmation (Recommended)" — over "the first anchor whose pattern confirms" (the bound could move later) |
| RX1-K5 | The two references at EXACTLY the same price → the STRUCTURE reference applies (its bound: the reversal pattern / the cross-limit watch's break) (PLAN_K K-Q5) | "Structure reference" — over the previous cycle's CTS (recommended: its pattern broke that price first, the stricter bound) |
| RX1-K6 | The watch detects patterns by MS's own convention — its look-ahead, `continuous` first — and a break counts once its confirmation has happened; the in-flight answer = the final answer cut by time, the bound never moves (PLAN_K K-Q6) | "MS's convention (Recommended)" — over a strict live view (a frame cut at the moment; the break's anchor could move) |
| RX1-K7 | A reversal-started structure's reversal pattern (the bound) is known for its WHOLE MS run — like `P` and the reference price (the structure exists only from its reversal; its earlier cycles are retroactive) (PLAN_K K-Q7) | "Whole run, like P (Recommended)" |
| RX1-C1 | Chart aid for reviewing RX1: the fib HOVER + the CSVs only (no reference line, no break marker) — the hover gives the reference price, its source and the break; `*_ip_candidates.csv` gives the cut candidates their reasons | AskUserQuestion: "Hover + CSV only" (over a dotted reference line + a break marker, recommended) |
| RX1-E | The condition is evaluated at IC SELECTION only (like §3.1), on the final fib for the drawn POI — not re-checked in the activation sweep (it cannot flip: the references are fixed per fib record; before its break it passes, after it the pick is already before it); MS's in-flight read evaluates it on its as-of fib at each read | "At IC selection only (Recommended)" |
| RX1-4 | No reference at all (H1 sid 0's and sub 0's cycle 0 — no cross limit, no previous cycle) → auto-pass (RX1-B) | "In cases where neither reference prices (prev cycle CTS or prev struct BOS / equivalent) are available, then we count these requirements as auto pass" (Q4: "I explained this in detail in a previous response") |
| RX1-F | **The IC flow (2026-10-09, PLAN_K landing A's review, question B9):** find IC candidates → find the IP among the IC candidates → fallbacks when either step fails. The IC conditions ALL belong to the first step and are processed TOGETHER to filter candles before the IP: (1) opposite-direction candle, (2) the reference price conditions (RX1), (3) an unfilled imbalance after the candle, (4) overlap with the 61.8–80 % fib zone. A candle failing ANY one is disqualified; an **IC candidate** passes all four. "No IC candidates" can NEVER be attributed to one condition (they filter in tandem) — so no reason, label or hover may say a single condition (e.g. the reference) "cut all the candidates" | "From a big picture perspective, our IC flow is: Find IC candidates > Find IP amongst IC candidates > Additional fallbacks when either previous step/task fails. The existing IC conditions as well as the new reference condition all belong collectively to the find IC candidates step/task, and they are processed together to filter candles before we look for IP. … If any of them fails, then they disqualify a specific candle. If we find no IC candidates, then that means these conditions disqualified all candles. So when there's no IP, it could be because we have IC candidates, but no pattern OR we simply didn't have IC candidates to begin with. Your concept that the IC reference condition specifically cuts all candidates is erroneous. The reason it seems like that is because we built the other conditions first, and now we are layering on another … in terms of actual processing, you can't attribute no IC candidates to just this new condition because all the other conditions also work in tandem to filter candles. This is a very important nuance in terms of how we build this logic and process the steps/tasks." "Down the line I will provide more detailed thoughts around how the holistic IC & POI process works." |
| RX1-N | **The "IP: none" reasons:** `no_ic_candidate` (NEW — no candle passes all four conditions) vs `no_pattern` (IC candidates exist, none forms an institutional pattern). Measured (S_K_A and V alike): confluence sub 1 cyc 2 (1934 → 1945; 3 candles pass conditions 1 + 3, none touches the zone) is today labelled `no_pattern` → becomes `no_ic_candidate`; the other 7 none records have IC candidates, none patterned (`no_pattern` stays); today's `no_candidates` (nothing passes 1 + 3) occurs nowhere | "I think the right new reason should be "no_ic_candidate", which is the real reason. As to why there are no_ic_candidate, you can't attribute that to just 1 of the 4 conditions. But this new reason should distinguish between having IC candidates but no pattern as opposed to not having IC candidates at all." Follow-ups (AskUserQuestion, 2026-10-09): today's `no_candidates` (nothing passes direction + imbalance) is folded into `no_ic_candidate` (my reading of RX1-F, stated, not objected); the relabel lands IN landing B ("Both in landing B" — over B first / relabel after, recommended): B's `/compare` vs S_K_A == V + sub 1 cyc 2's relabel; the docs / names: "Wait for your write-up" (over rewriting POI_ZONES_SPEC §3 now, recommended) — only the reason label changes now, §3's step structure and the code's "candidates" identifiers wait for the user's holistic IC & POI write-up; the fib's BOS → CTS window = the SCOPE the four conditions are applied in, not a fifth condition (recommended) |

**2026-10-06c (the revamp session opens with B's landing review — §8.5c)**

| ID | Decision | Why / the user's words |
|---|---|---|
| LR1 | Every landing-review finding fixed in ONE fix-up commit: the lens's 8 killing tests (the vacuous M15 sub-POI source pin replaced by a rendered lens test), a `unified_probe` unstamped-frame test (C2), the doc fixes C1 / C3–C9, the C10 log counter, a review mutants file re-run to all-killed (B29 repaired); `/compare` vs S_F1, predicted byte-identical, no save | AskUserQuestion: "Fix all, one commit (Recommended)" |
| LR2 | C1 guarded by a STATIC test, not only by docs: no `except` clause in the engine modules catches `RuntimeError` / `Exception` / bare, except an allowlist; the docs name `RuntimeError` | "Static guard test (Recommended)" |
| LR3 | The cross-cycle fib revamp (register §D.1 item 2) runs as THREE sessions: this one = the user's rules → the spec (+ cheap status-quo measurements); the next = the implementation plan from a fresh code audit + a cold review; then the build | "Three sessions (Recommended)" — closes F2c |

**2026-10-06 (the chart-debugging session PAUSED — the cross-cycle fib revamp next)**

| ID | Decision | Why / the user's words |
|---|---|---|
| F2a | The IP / IC / POI debugging is PAUSED; every logged point stays as it is (§8.5b C2 / T1 / C3 / H1 / HV, the §8.9 open item sub 3 cyc 3) and is picked up after the cross-cycle fib revamp, compared against S_B and S_A as planned | "After looking at the Fib levels and chart outputs, I would like to put additional debugging on IP, IC & POI on hold for now." … "we can park the debugging and come back to it after the cross-cycle fib update. Any items we've logged regarding debugging we will come back to later." |
| F2b | NEXT: the user revamps the cross-cycle fib rules — one logic for main and sub, meant as the final cross-cycle + single-cycle fib logic. It replaces register §D.1 item 2's open decision (single-cycle fib POIs on subs) and runs before item 1's debugging resumes | "since we are now reviewing Fib levels, it would be a good time to work on the parked "Decide together with the parked "single-cycle fib POIs on subs." But instead of deciding on this, I will instead revamp the cross cycle fib rules. Furthermore, my hope is that we can unify the logic across main and sub. This update will likely be the final logic we use for cross-cycle & single-cycle fibs." |
| F2c | The revamp runs in a NEW session, which also decides whether planning and build become separate sessions | "For the cross-cycle fib update, I think it's probably better we start a new session. In the new session, we can also decide if we need to separate planning and build sessions." |
| F2d | `/commit-save` now, after the F1·1 hover fix — the baseline the revamp is compared against | "I think we can do a commit-save here so we have a comparison with the next iteration … The commit-save now will also help us compare the updated fib logic vs. what we have now."; the `%%` fix: "Fix before the save" |
| F2e | B's landing review (B `e54f509` + F1, ~0.55M, 2 read-only lenses) runs FIRST in the revamp session, before its plan — was: deferred until the debugging rules settle (§8.5a). The branch stays local | "Next session, first"; push: "Keep it local" |

**2026-10-06 (the chart-debugging session, §8.9)**

| ID | Decision | Why / the user's words |
|---|---|---|
| F1a | The M15 lens charts draw each sub's fib records (they were CSV-only) + the fib hover's `IP: none (<reason>)` — a chart aid, its own commit | "Add the IP: none reason to the M15 fib hover first … currently are also not charting the fibs on the sub charts (I think currently they are set to hidden) … could we also add the charting of the Fibs on the M15 subs so it helps with debugging and reviewing?" |
| F1b | The fib draw gate: not collapsed AND (locked OR active), ended or not — a collapsed cycle's fib is hidden like its KL / POI; EVERY ended never-locked record is drawn (any end reason; on the window only sub 3 cyc 3, `parent_end`); `active` False (a superseded cross version, a fib invalidated by an imbalance fill) still vanishes | "we do not need to chart any degenerate Fibs … these degenerate fibs really don't provide value because they cannot be traded in live scenario. We do chart the locked, or active with cycle not ended. In addition, I would also like to chart the never-locked records that were ended by parent_end" → "Every ended record"; the `active` False reading: "Yes, they vanish" |
| F1c | "Collapsed" = what the user calls degenerate: ONE test (a cycle whose clamped lifecycle start ≥ its end — never live) under two code names — "collapsed" on the zone / chart side (any structure), "degenerate parent cycle" in `ParentTables` (H1 parent cycles, M15 coordinates; triggers inside are rejected). On H1 the same two cycles (1,0) / (1,1). The code's names kept; a note for the parked Naming-Standard audit | the user asked "what do you mean by collapsed fib? … does that mean these 2 are not the same idea? How are you defining degenerate then?" — answered: the same idea, two names (my first answer had overstated the difference) |
| F1d | The H1 chart follows the same gate: its collapsed fib sid 1 cyc 1 (689 → 761) is hidden | "Hide it on H1 too" |
| F1e | One legend toggle "M15 fib" per lens chart | "One toggle" (two — collapsed separately — were moot once collapsed fibs are hidden) |
| F1f | Fib hover extras on both charts (one builder): the pick `IP <rank> · IC <idx>` and the lifecycle `life=[start, end] reason` | "The pick, The lifecycle" |

**2026-10-06 (the plan review's questions, §8.8 — each answered with the recommended option)**

| ID | Decision | Why / the user's words |
|---|---|---|
| R1 | `first_active` takes a MOMENT, `pattern_confirmed_idx` = the continuation's last candle when the leg hit the cap, else the next candle (the one that shows the leg ended, after any trimmed dojis) — refines D1b; conf sub 2 cyc 0: IC 2383, continuation 2384–2385 → confirmed 2386 | "Use the moment" |
| R2 | Names: `ip_leg_candle_counts`, `opp_leg_first_idx`, `ip_tb_imbalance_first_c2_idx`, `ip_tb_runner_up_ic_idx` ("size" = a price distance; `start_idx` / `end_idx` = lifecycle; say which candle) | "Rename all four" |
| R3 | B's cond5 reason `ic_not_touching_fib_zone` (refines D1a; beside cond1's `ic_out_of_fib_bounds`) | "ic_not_touching_fib_zone" |
| R4 | Landing A ends with `/commit-save` (local); B's `/compare` against A's save (A's own names the Week 8 save) | "/commit-save after A" |

**2026-10-05 (round 6 answers)**

| ID | Decision | Why / the user's words |
|---|---|---|
| D3c+ | Cap 1 is ONLY for 3D's initial leg; a junction doji placed in the initial leg may stack it to a maru and keep the opp leg single (J1) | "Cap 1 is for 3D only and is ONLY for 3D's initial leg" / "If we have a big_normal, doji, then single candle opp candle. We can have a scenario where doji joins initial and forms a stacked maru, and that allows the opp run to be a single" |
| E1-confirm | The trim also covers a cap cut (supersedes P6d): the doji counts toward the cap, then is dropped — ↑ ↑ ⊖ ↑ ↑ · opp → initial ↑ ↑; conf sub 4 cyc 0: 3464 ↓ 3465 ↓ 3466 ⊖ \| 3467 ↓ → 3464–3465 | "Yes your understanding and reading are correct" |
| HV | POI hover: `IP <rank> · legs a-b-c (initial \| IC \| continuation idx)` + a tiebreak line when TB1 / TB2 decided (`TB2 over <idx> (reference imbalance <c2>)`); fib hover "IP: none" (C1) | "Sounds good" |
| L1 | Two landings: A = the IP selector + the debug CSVs, nothing switched (`/compare` byte-identical but the new CSVs); B = the switch (drawn + MS, the rule-3 `versions` migration, HV) against a prediction from A's CSVs | "Ok sounds good" |
| K1 | Checkpoint commit of the spec docs (local, docs only, no push) | "Ok to commit" |

**2026-10-05 (round 5 answers)**

| ID | Decision | Why / the user's words |
|---|---|---|
| D3c | The 3D initial leg has cap 1: the single big_normal candle right before the opp leg, never a stack; evaluated next to the cap-3 initial, the best pattern wins; a true doji joins the 3D initial only if that yields something better than 3D | "if a leg has to be single, in essence it's similar to saying that the cap is set at 1 … if the initial leg in of itself is a big_normal, then it has already met the 'cap' … the doji cannot join initial and must join opp" / "the initial leg must be just the single big_normal. We cannot 'stack' any candles to achieve a 'stacked' big_normal to count as 3D. However, it doesn't mean we cannot use the big_normal candle as a part of a stack for a multi initial leg if we can achieve a higher ranked pattern that's not 3D" / "if keeping a doji … in the 3D initial leg (and combining it with other previous candles) ultimately changes the pattern to something higher than 3D, then that is acceptable. Otherwise … it goes in opp" |
| E1 | A true doji never starts or ends the pattern: never the initial leg's first candle nor the continuation leg's last (trimmed) — the cap-cut case → E1-confirm | "a true doji should never start or end the pattern … it should not be the first candle of an initial leg or the last candle of the continuation leg, even if say those legs are already multi and their caps have not been met" |
| X1 = keep | `select_ic_variants` + `POIConfig.v30 / v60 / v90` stay in the code (unwired, with their tests) as a possible no-IP fallback (tbd; register §D.2 next to H2) | "I'd like to keep them for now. There is still a chance that I use those as fallbacks in the future for instances where no IP can be found … we might still delete in the future" |
| N5 | agreed direction / true doji / volume-only doji / split candle | "yup sounds good" |
| Q-W note | IP only now; wave candles + WVMI likely follow later (register §D.2 line) | "we likely will also make the change for wave candles & WVMI, but we'll come back to that later. Let's make a note" |
| C3 | The proximity-scan cascade (§3 round 5) is expected; judged at the chart review | "these would all be expected changes if POI zones move. We won't be able to judge whether those changes are appropriate or correct until I can review the charts" |

**2026-10-05 (round 4 answers)**

| ID | Decision | Why / the user's words |
|---|---|---|
| P6c-final | True dojis are full members of the leg they are ASSIGNED to (cap + multi); the assignment is what matters. ↑ ↑ ↑ ↓ ⊖ ↑ ↑: the doji cannot make the opp leg multi (the IC is never a doji) → it joins the continuation and makes it multi; a 3D initial must stay single → a doji after it joins opp and makes opp multi | "dojis do count toward the cap and make the legs that they are assigned to multi. We simply need to assign them to the right legs. If they get assigned to the wrong legs, then they may erroneously make a single leg multi" (replaces the round-3 recommendation "cap yes, multi no") |
| J1 = a | A true doji between the initial and opp legs is tried in each leg, each under that leg's cap rules (3 where multi is allowed — the window slides, pushing the earliest candle out; 1 where only single is allowed, 3D), and the best pattern wins; rule B's reason stands (a single opp leg is stronger) | "even if the initial leg is above the cap, then it may still be more preferable to have doji join initial if it would keep opp as a single candle, and instead cut off 1 earlier candle that would have been part of initial … try the doji in each leg to then select the best available pattern" / maru · ⊖ · maru(IC) · big_maru: "maru-maru-big_maru (multi-single-single) … is the stronger pattern compared to … maru-multi-big_maru (single-multi-single)" / Ex 1: "3D simply says that the initial leg cannot be a stacked big_normal, but a big_normal single candle could still be part of a multi leg that forms a maru or big_maru" (the cap-1 reading → D3c) |
| VD | A volume-only doji is a directional candle: in the leg of its direction, and it CAN be the IC (the last opp candle); the doji rules apply only to true dojis → §3.1's direction condition reads the agreed direction | "A doji that still has a direction is not one that can count toward either leg. In this case, [it] should be part of the opp leg, and would be the IC … the doji rules around which leg they should go to only apply to true dojis" |
| TB2a | The reference = the oldest unfilled imbalance after the oldest tied IC; the pick = the most recent tied IC before it; an IC may open the gap (c1, also of a multi-candle gap); "unfilled" = §3.1's test | "Yes 4083 should be the correct … the reference unfilled imbalance should be 4132 and IC should be 4083" / "if 4029, 4132 did not exist … 4151 would be the reference … but the IC would be 4136" / "only possible in the scenario where the candle that opens the gap is in the opposite direction since IC must be opposite direction candles" |
| Q-W = a | The agreed direction for the IP only; wave candles / WVMI keep "qualified candle" (register §D.2 line) | "for now I think we should just apply to IP" |
| T1 | The touch stays on the IC: an opp leg whose earlier candle touches but whose IC does not is no pattern | "based on the rule we have now … it would not be a valid pattern. However, depending how charts look I may ultimately adjust the rules, the pattern, and how we identify IC, but that's tbd" |
| C2 | The no-POI cycles: revisit at the chart review | "we will need to come back and understand if losing their POIs in these cycles is appropriate" |

**2026-10-02 (round 3 answers)**

| ID | Decision | Why / the user's words |
|---|---|---|
| V1a | The agreed direction (§1 item 2): candle and volume direction must agree; when one is 0 the other decides; both 0 = a true doji; opposite = breaks (V1) | "we should treat consistently volume direction 0 and candle direction 0 (doji) … when one of them is 0, we can simply take the direction of the other … The doji rule we discussed earlier (not breaking legs & also when in between legs) should only be applied when both volume & candle direction are 0" |
| P6b | A true doji counts toward the cap: IC · ⊖ ↑ ↑ ↑ → the continuation is ⊖ ↑ ↑ | "because it counts as part of the up" |
| P6c | A true doji in a leg makes it multi (opp and continuation); a doji between opp and continuation is never part of opp (↑ ↑ ↑ ↓ ⊖ ↑ ↑ → single opp); a 3D initial stays single → a doji after a single up joins opp (general rule → J1); full-member reading → P6c-confirm | "If the doji is counted as part of opp, then it is multi … The doji can never be the IC, so it does not count as part of opp, which means there is only the down, so it is a single" / "On continuation, multi should be allowed so it's like everywhere else?" / "after the direction edits, there won't be that many true dojis" — → P6c-final (2026-10-05) |
| P6d | A pattern may start (or end) on a true doji | "yes allowed" |
| P6e | Superseded by V1a: a direction-0 candle with a volume direction is not a doji, it takes that direction | "See what I said above" |
| TB2 | Tiebreakers: 1 IC maru > normal > pinbar; 2 the most recent IC before the oldest (earliest) unfilled imbalance — replaces the round-2 list (length / body bands, more recent); TB-a / TB-b moot | "Let's change the rule again" (reading → TB2a) |
| D2 | Confirmed as answered | "ok" |
| D5 | The sub snapshot's fib divergence is fixed in item 2 | "I think this IS something we do need to fix, but we can do that as item 2" |
| C1 | Debug visibility: a per-lens `*_ip_candidates.csv` (every candidate: legs, rank or the failure reason) + a fib hover line "IP: none" for no-pattern cycles | "Ok sounds good" |

**2026-10-02 (round 2 answers)**

| ID | Decision | Why / the user's words |
|---|---|---|
| V1 | Every leg candle needs candle direction AND volume direction with its leg (initial / continuation: sd; opp: −sd), single and multi; a candle failing it breaks the run | "This is to account for candle gaps between candles that cause volume direction to be different than candle direction … For multi, it could mean some consecutive direction candles are broken. This is an important change to get to correctness." |
| ~~TB~~ | ~~Tiebreakers: 1 IC maru > normal > pinbar; 2 larger total length (within 80 % = tie, a param); 3 larger body (within 80 % = tie, a param); 4 the more recent candle~~ — superseded 2026-10-02 by TB2 (it had superseded R4 / R6) | "will potentially try other values down the line so should be a param" |
| P2 / item 5 | A stacked candle is defined exactly like a single candle: the same classifier and the same parameters (types and big flags) | "the way we define a big 'stacked' candle is the same as how we would define from a single candle perspective. It should be using the same parameters" |
| D2 | Confirmed: after the update the drawn POI and MS's snapshot must not pick different candles | "they should not be picking new candles after the new logic update" |
| D4 | Keep POI_ZONES_SPEC §3.2 in the spec, unapplied, for future testing; IC conditions stay outside the pattern, the pattern is applied last | "These were potential logic I would have implemented. I still want to keep these and test into them in the future … it makes sense to keep all IC related logic outside the institutional pattern, and apply patterns last after all other IC conditions" |
| H2 | The lower-TF fallback → one line in register §D.2 | "sounds good" |
| P6a | Doji junctions: initial↔opp → the initial leg unless it is at the cap (then opp); opp↔continuation → always the continuation | "opp has no cap, and doji does not alter body of opp which is what's compared to continuation" / "we do not want doji as the IC, so we never want doji as the last candle of the opp" |
| B1a | A run crossing the BOS / CTS anchor is cut at the anchor | "cutting is fine" |
| R4a | TB2's parenthetical was the touch rule restated (moot after TB) | yes |
| D1a / D1b | Re-check = touches the as-of zone (`ic_outside_fib_zone`); `pattern_end_idx` in `first_active` — refined 2026-10-06: the reason `ic_not_touching_fib_zone` (R3), the moment `pattern_confirmed_idx` (R1) | ok |
| P8a | 3D last; single initial = a 1-candle run; 2B/2C/3B keep multi initials; no type requirement on a multi opp leg | "we are always taking 1 candle of the stack as the IC which is always weaker than if the IC was 1 true single candle" |

**2026-10-01 (round 1 answers)**

| ID | Decision | Why / the user's words |
|---|---|---|
| D2 | ONE selection function feeds both the drawn POI and MS's sd-proximity read | "yes" — accepting the H1 sid 1 cyc 2 cascade (§3) |
| D1 | Today's shape: pick on the final fib; the activation sweep gates when it is live | "From an entries decision perspective, the zones do not matter until the price retraces back toward the zone … every CTS event (established, updated) means we are still forming new highs, so price would not be retracing. When CTS becomes confirmed (via pullback or proximity), is when we know price is retracing toward our zones" |
| B2 | The continuation leg never runs past the CTS anchor | "should be very rare … the candles would have to be massive. But just in case, I would say no" |
| B1 | The initial leg never starts before the BOS anchor | "consistent with CTS" |
| P4 | Each leg = the MAX consecutive run available, capped at 3 (initial: the most recent 3; continuation: the first 3) | "5 consecutive sd candles → the most recent 3 … only 1 sd candle, and immediately before that a −sd candle → 1 … max consecutive available, up to 3, and the same for continuation wave" |
| P3 | big_maru / big_normal measured against the 5 marus before the WHOLE pattern's first candle | "should be the same as other candle patterns" (amends the spec's "prior to the first candle in the wave segment") |
| R1 | Letter-rank first, then shape (1A.iv > 1B.i) | "Yes correct" |
| ~~R4~~ | ~~TB2 by price: the IC's OUTER bound closest to BOS~~ — superseded 2026-10-02 by TB | (was: the outer bound is the rough stop-loss reference) |
| P1 | Single-candle types = the existing `candle_type` | yes |
| P5 | Opp leg = the whole run of consecutive opposite candles ending at the IC | "all must be consecutive in the same direction" |
| P6 | Direction-0 candles do not break any leg | details → P6a |
| P7 | No extra pattern requirements | an imbalance formed by the continuation leg = extra strength → the later zone-strength topic (register D.2) |
| P8 | Unlisted combinations → no pattern; **3D added**: big_normal (single only) - multi - big_maru, shapes i s-m-s / ii s-m-m | "we only allow single candle on initial leg because that is already an exception, allowing multi would be even weaker" |
| P9 / P10 | POI = the IC's high / low; body ">" strict, touch inclusive | yes |
| R2 / R3 / R5 / ~~R6~~ | best per IC; TB1 separates only multi-opp ranks; TB1–TB3 on the IC candle; ~~exact tie → earlier idx~~ (superseded by TB4 the more recent) | yes (R2 is moot under P4: one leg triple per IC) |
| N1 / N3 | initial / opp / continuation leg; `ip_rank` | yes |
| N2 | big_normal = type normal + the big_normal ratio; the end candle = type maru + the big_maru ratio | "you are correct" |
| N4 | "earliest" in the spec was a typo for "most recent" | — |
| D3 | The same IC may serve several cross-cycle cycles | yes |
| D6 | `ip_rank`, leg sizes, leg idx ranges replace `versions` / `current_versions`; reason rename; chart: hover only | "Hover is fine for now" |
| D7 | No pattern → no POI | ok |
| H1 | The leg cap as a per-TF table (H1 3, M15 3; M5 2 later) | "fine for now" |
| H3 | The spec + figures live in `plan_h_inputs/`, amended in place | "Feel free to edit the copy based on our discussion" |

## 7. Change log

- 2026-10-01 — created from the user's spec + the /prepare findings; H1 prototype round 1.
- 2026-10-01 — round-1 answers recorded (§6); spec copied to `plan_h_inputs/` and amended (N1 naming header, N2, N4,
  P3, P4, P5, P6, B1/B2, 3D, R4, R6); §1 rewritten as the engineering reading; H1 prototype round 2 (§3); round-2
  questions (§5).
- 2026-10-02 — round-2 answers recorded (§6): the volume-direction leg rule (V1), the new tiebreakers (TB; R4 / R6
  superseded), the doji junction rules, P2 / item 5, D2 / D4 / H2, B1a / R4a / D1a / D1b / P8a; spec amended; §1
  items 2, 5–7, 10, 13, 14; H1 prototype round 3 (§3: picks unchanged); round-3 questions (§5).
- 2026-10-02 — round-3 answers recorded (§6: V1a the agreed direction, P6b–P6e, TB2, D2, D5, C1); spec amended; §1
  items 2, 3, 5, 10; prototype round 4 on H1 + both M15 lenses (§3: H1 unchanged; M15 3 same / 12 different / 5 none;
  TB exercised twice; 1902 likely survives); round-4 questions (§5: P6c-confirm, J1, VD, TB2a, Q-W, N5).
- 2026-10-05 — round-4 answers recorded (§6: P6c-final, J1 = a, VD, TB2a, Q-W = a, T1, C2); spec amended (VD, P6 /
  J1, TB2a, 3D note, candidates); §1 items 1–5, 9, 10, new 15 (who reads the POI); prototype round 5 (§3: picks
  unchanged under every switch; VD adds / drops candidates without a pick change; the proximity-scan cascade measured
  — 710 → 748 and 926 → 939 move the WVMI gate and the counter subs' triggers, 6 of 11 trigger records re-derived);
  round-5 questions (§5: D3c, E1, X1, N5).
- 2026-10-05 — round-5 answers recorded (§6: D3c cap 1, E1 trim, X1 keep, N5, the Q-W note, C3); spec amended (E1,
  D3c, X1); §1 items 2, 4, 5, 9, §4; prototype round 6 (§3: picks unchanged; E1 trims one real leg — a cap cut;
  D3c changes no best); round-6 items (§5: E1-confirm, HV, L1).
- 2026-10-05 — round-6 answers recorded (§6: D3c+, E1-confirm, HV, L1, K1); §1 item 9 (cap 1 only for 3D's initial;
  an opp run is never cut to its IC vs a junction doji's placement); status SPEC DECIDED; checkpoint: the inbox filed
  (register §D.1 / §D.2 / §A, POI_ZONES_SPEC, CROSS_CYCLE_FIB_SPEC, memory) and PLAN_H + `plan_h_inputs/` committed.
- 2026-10-05 — §8 the implementation plan DRAFTED (design, the audit table, landings A / B with their `/compare`
  predictions, the shadow between them, tests, size and cost); awaiting its review.
- 2026-10-06 — §8 cold-reviewed (3 read-only lenses, 879,656 tokens): §8.8 records every finding (PR1–PR14) with
  its disposition; R1–R4 asked of the user (the pattern's knowability, field names, the cond5 reason name, A's save).
- 2026-10-06 — R1–R4 answered (§6: the moment `pattern_confirmed_idx`, the four renames, `ic_not_touching_fib_zone`,
  `/commit-save` after A); §8.0–8.7 rewritten with every §8.8 row folded; §1 item 14, §2, §4; status: NEXT landing A.
- 2026-10-06 — landing A cold-reviewed (2 lenses, 453,157 tokens; §8.3a): 0 BLOCKER; 22 test gaps + 3 fail-loud nits
  folded (tests 1219, mutants 98/98 KILLED); two B items into §8.5 step 3 (the sub-geometry `except`, the cost).
- 2026-10-06 — landing A BUILT (§8.3a: code, 4 deviations, 89 + 2 tests → 1198, 66/66 mutants KILLED); `/compare` vs the
  Week 8 save == the prediction (24/24 md5, figures + hovers equal, 3 new CSVs); A's check vs the prototype: every
  record equal but the gated-out sub 3 cyc 3; docs: POI_ZONES_SPEC §3.4, GLOSSARY, IMBALANCE_FILL_SEMANTICS, the
  `/compare` skill + WORKFLOWS + review_scripts README (27 CSVs).
- 2026-10-06 — the chart-debugging session opened (§8.9; decisions §6 F1a–F1f): F1 the M15 fib layer + the fib
  hover's lifecycle / IP outcome on both charts (a chart aid; the 27 CSVs byte-identical to S_B).
- 2026-10-06 — F1·1 (§8.9): the fib hovers' literal `61.8%%` / `80%%` → `%` on both charts (CSVs byte-identical).
- 2026-10-06 — the debugging session PAUSED (§6 F2a–F2e): the cross-cycle fib revamp next (register §D.1 item 2, a
  new session opened by B's landing review); `/commit-save` = the revamp's baseline.
- 2026-10-09 — the debugging session RESUMED after the revamp (PLAN_I S1–S6 + PLAN_J): §8.9 R0 = every logged point
  re-derived on S_S6 (PLAN_I §9.9 confirmed; C3 + the H1 points unchanged; sub 0 cyc 2 gains a POI, sub 2 cyc 1 and the
  new H1 (0,0) have none; sub 3 cyc 2's pick 2969 TB1, touch-delayed to 3044).

## 8. Implementation plan (2026-10-05; REVIEWED + folded 2026-10-06 — §8.8; A built + saved §8.3a; §8.4 measured §8.4a; B built + saved §8.5a, chart-review points §8.5b)

### 8.0 Shape

Two landings (L1).
- **A** builds the selector and exports what it would pick, switching nothing: tests → replay (timing shown) →
  `/compare` against the Week 8 save named explicitly (`artifacts/commits/week8-volmom-multitf/20260930_151531_8b11bc2`,
  the TRAP; `cmp_save.py --names-from <that save>` — there is no `LATEST_week9-entries-decisions` yet) → A's check vs
  the prototype → the test census (§8.4) → **`/commit-save`** (save S_A, local; R4).
- **Between them**, three variant replays write B's prediction (§8.4).
- **B** switches the drawn POI and MS's live read together (D2) and migrates the POI meta (rule 3, one commit, the
  prediction in its message): tests → replay → `/compare` against S_A (== V3 + the format-only deltas) → chart review.

Byte-identity is claimed only with `cmp_save` + `cmp_hover` + `cmp_fig_full`.

### 8.1 Design

- **`features/candles_v2.agreed_direction(direction, vol_dir) -> np.ndarray`** — the V1a table: ±1 agreed, 0 a true
  doji, `AGREED_SPLIT` (a module constant ≠ ±1 / 0) a split candle. A pure function, NOT a df column (no `_final.csv`
  change). ONE helper (§1 item 2): `wave_candles._is_qualified` and the inline check in `wvmi._find_temporary_lp` become
  `direction == d and agreed == d` (== today's `direction == d and vol_dir in (d, 0)` for d = ±1 — byte-identical;
  their behaviour is unchanged, Q-W) (PR10).
- **Per-frame candle params.** `apply_candle_classification` stamps `df.attrs["timeframe"]` and
  `df.attrs["candle_params"]` (the params it classified with); `candle_classifier.candle_params_for_timeframe(tf)` is the
  one reader of `DEFAULT_PINBAR_BODY_PIP_FLOOR_BY_TF` and keeps today's `.get(tf, 0.0)` (byte-identical). No re-stamp
  at orchestrator ~:928 / :934: the attrs survive every frame the IP sees (measured — the main frame, the shared M15
  frame, the lens copies, the bounded sub slices after `reset_index` + `compute_imbalance`, the MS resolver frame, the
  probe's Phase-2 frame) → a test pins them. The IP reads BOTH from the frame it is handed — so a stack is classified
  with exactly that frame's single-candle params (P2), also inside the probe's Phase-2 MS, whose MS `timeframe`
  defaults to H1. Missing → `KeyError`.
- **`candles_v2._big_flags(length, prior_max, params)`** — the single-candle big-flag rule (ratio, thresholds, the
  `prior_max <= EPS` → no flag guard) factored out of `classify_big_flags`, byte-identical; the stacks call it (PR9).
- **`patterns/imbalance.iter_unfilled_imbalances(...)`** — the body of `has_unfilled_imbalance` as a generator (same
  arguments, filters and order); `has_unfilled_imbalance` = `any(...)`, byte-identical (checked by the reviewers).
- **`patterns/institutional_pattern.py` (new — the pattern only: no fib, no candidate rules, D4).** Implements §1
  items 3–9 in three separable steps, so `select_ip_ic` classifies every candidate's stacks in ONE `classify_candles`
  call (PR8; `attrs["pair"]` copied for the pip size):
  1. `leg_variants(df, ic_idx, sd, *, bos_anchor_idx, cts_anchor_idx, agreed, cap)` — the last-opp check (true dojis
     after the IC belong to the continuation; the scan bounded at `cts_anchor_idx`); the continuation run (cut at the
     CTS anchor, the cap, the E1 trim, ≥ 1 sd candle); the whole opp run (P5, never cut) and its leading true dojis =
     the junction; the initial run (cut at the BOS anchor); for every junction split, the cap-`cap` initial (E1-trimmed,
     ≥ 1 sd candle after the slide) AND the 3D cap-1 initial (D3c). Each variant = index ranges.
  2. The stacks of all variants → one classify call; big flags via `_big_flags` against `prior_maru_max_len_as0` at
     each variant's `pattern_anchor_idx`.
  3. `rank_variant(...)` — the rank / shape table as ONE data constant (a chart-review re-rank is a one-line change);
     the continuation big_maru with a body strictly greater than the opp leg's; best (rank, shape); a full tie →
     rule A / B.
  `classify_ip(df, ic_idx, sd, ...)` composes the three for one IC (the tests). `IPResult` (frozen): `ic_idx`, `rank`,
  `shape`, `pattern_anchor_idx`, `opp_leg_first_idx`, `pattern_end_idx`, `pattern_confirmed_idx`,
  `ip_leg_candle_counts` ("a-b-c"), `reason` (None for a pattern), `detail` (stack types / ratios / bodies for the
  CSV). **`pattern_confirmed_idx` (R1)** = the moment the continuation leg is known complete: with run = the
  continuation run up to the cap BEFORE the E1 trim, `run[-1]` if `len(run) == cap`, else `run[-1] + 1` (the candle
  that shows the leg ended — also for a run cut at the CTS anchor); past the frame's last candle → the POI is never
  active (the sweep never reaches it). Only the activation reads it; MS's read is gated after the CTS anchor.
- **`zones/poi_zones.py`:**
  - `POIConfig.ip_leg_cap_by_tf` = `field(default_factory=lambda: {"H1": 3, "M15": 3})` (unknown TF → `KeyError`; M5
    2 with the M5 lens). `v30 / v60 / v90` stay (X1).
  - `find_ic_candidates(..., ic_direction=None)`: `None` = candle direction (the kept V path, unchanged); the IP passes
    the agreed direction (VD).
  - ONE zone helper `fib_zone_bounds(...)` (61.8–80 from the fib's prices) + `ic_touches_zone(h, l, zone_lo,
    zone_hi)` (inclusive) — the selection AND B's activation re-check read both (exact edges, float).
  - `select_ip_ic(df, fib_state, config, *, classify_misses=False) -> IPSelection(candidates, pick, tiebreak)` —
    §3.1 candidates (agreed direction) → touch → `leg_variants` / stacks / rank → best (rank, shape) → TB1 → TB2a.
    `classify_misses=True` (the CSV only) also classifies the candidates that miss the touch (`touches_zone` False,
    reason `no_touch`, never picked — debug only; the T1 chart-review cases 2375 / 4048) (PR11).
  - **TB2a**: reference = the min-`start_idx` instance of `iter_unfilled_imbalances(df, oldest_tied_ic_idx + 1,
    cts_anchor_idx, cts_anchor_idx, fill, direction=sd, evaluated_at=None)` (cond3's own test, TB2a);
    `assert reference.start_idx > oldest_tied_ic_idx` (structural — a c2's candle direction is sd, an IC's never is;
    PR13); pick = the most recent tied IC with `ic_idx < reference.start_idx` (an IC may be c1). `tiebreak` =
    `(rule, runner_up_ic_idx, tied_count, imbalance_first_c2_idx)` or None.
  - `_poi_fib_records(fib_states, structure_events)` — today's per-record gate (active and not ended, or locked) AND
    `derive_poi_zones`' "no CTS_ESTABLISHED → skip" (`poi_zones.py:496`), so A's CSV rows == B's POI records (PR11).
  - One parameter source: the orchestrator builds `POIConfig()` (its re-typed literals go — all 6 fields equal,
    measured); the resolver keeps `POIConfig(fill_threshold=...)` (MS's own threshold, unchanged).
  - The MS resolver (B) re-picks at every CTS_ESTABLISHED / CTS_UPDATED on the AS-OF fib, the continuation cut at the
    as-of CTS — an as-of pick can differ from the final drawn pick (D2 pins the final fib; V3 counts them) (PR12).
- **The candidate records** (`IPCandidateRecord`, a dataclass): the `IPResult` fields + `touches_zone`, `picked`, the
  tiebreak fields and an attribution `meta` (`structure_path_id`, `sub_id`, …, like zones); IP data lives in dataclass
  fields, not `*_idx` dict literals in `poi_zones.py` (the static idx-key scan, PR4); every index cast with `int()`
  (`_shift_meta_indices` skips `np.int64`); `__deepcopy__` → self (pandas deep-copies attrs on `__finalize__`; GOTCHAS,
  `common/types.py:147`). The IP never writes `inst.meta` (the imbalance CSV dumps it).

### 8.2 Audit table — every site that reads or writes the selection

| Site | Today | A | B |
|---|---|---|---|
| `poi_zones.find_ic_candidates` | §3.1, candle direction | + `ic_direction` (default unchanged); characterization pins (PR7) | the IP passes the agreed direction |
| `poi_zones.select_ic_variants` | V30 / V60 / V90 | unchanged; characterization pins (0 direct tests today) | unwired, kept with A's pins (X1) |
| `wave_candles._is_qualified`, `wvmi._find_temporary_lp` | the qualified rule, twice | call `agreed_direction` (byte-identical) | — |
| `candles_v2.classify_big_flags` | inline big-flag rule | `_big_flags` factored out (byte-identical) | — |
| `_run_downstream_pipeline` (main + every sub projection) | POIConfig literals; `derive_poi_zones` | `POIConfig()`; `select_ip_ic(classify_misses=True)` per gated record → `downstream["ip_candidates"]` | the picks are handed to `derive_poi_zones` (computed once) |
| `poi_zones.derive_poi_zones` | 1–3 POIs per fib record (one per variant pick) | unchanged (the gate via `_poi_fib_records`) | 0–1 POI per record = the pick; keeps an internal selection when none is passed (3 direct test callers) |
| `_compute_poi_activation_history` | cond5 = V30 overlap on the as-of fib; `first_active = max(moment, ic, floor)` | unchanged | cond5 = `ic_touches_zone` on the as-of zone, reason `ic_not_touching_fib_zone` (R3); `+ pattern_confirmed_idx` (R1); entries lose `versions` |
| `poi_zones.compute_poi_inners_for_cycle` (MS live read; every MS run: main, subs, probe Phase 2) | candidates → variants → inners, all inside `except Exception: return []` | unchanged | `select_ip_ic` → 0–1 inner, OUTSIDE the `try` (which keeps only the anchor selection + fib construction) (PR1) |
| orchestrator main attrs / `meta` | `poi_zones` | + `ip_candidates` | — |
| `multitf/types.LowerTFResult` + `entity_df_mutation.render_sub_projection` / the mirror | POIs shifted (`ic_idx`, `_ZONE_META_IDX_KEYS`, `activation_history`) | + `ip_candidates` (field, default empty); shifted by `_IP_CANDIDATE_IDX_FIELDS` (`ic_idx`, the fib anchors, `pattern_anchor_idx`, `opp_leg_first_idx`, `pattern_end_idx`, `pattern_confirmed_idx`, `ip_tb_runner_up_ic_idx`, `ip_tb_imbalance_first_c2_idx`) | `_ZONE_META_IDX_KEYS` + the same six POI keys |
| `tests/test_event_meta_idx_keys.py` | the shift guards | the static scans extended to `institutional_pattern.py`; the candidates' value pin | the new keys in `must_see`; `ip_tied_count` → `NON_INDEX_INT_META_KEYS`; a non-empty POI assert first |
| `zone_proximity.check_zone_proximity` | the nearest active POI inner | unchanged | code unchanged; reads the new POIs (§3 round-5 cascade) |
| `orchestrator._wvmi_trigger_streams_by_lens` (:724–750) | the main's first sd-prox trigger per cycle | unchanged | code unchanged; the confluence WVMI `triggered_by_*` cells move with 710 → 748 / 926 → 939 (PR3) |
| `debug/export_zones` + `run_replay` | `*_poi_zones.csv` (`versions` column) | + `export_ip_candidates` → `*_ip_candidates.csv` (H1 + each M15 lens) | `versions` column → `ip_rank` (a save-format boundary) |
| charts: `export_plotly` ~:2416, `export_m15_chart` ~:1623 (sub POIs) / ~:2531 (H1 overlay) | hover `versions=` | unchanged | ONE hover builder (`_zone_render`): `IP 1A.iv · legs 2-1-2 (4081–4082 \| 4083 \| 4084–4085)` + the TB line (HV); the H1 fib hover "IP: none" (M15 fibs are CSV-only) |
| charts: `_poi_confirmed_idx` (`export_plotly` :2076–2102, `export_m15_chart` :2637) | "multiple POIs per cycle (V30/V60/V90)" comments | unchanged | comments corrected; the not-found → −1 path kept |
| `POI_LIFECYCLE_DEBUG` dump (`derive_poi_zones`) | peak / current versions | unchanged | `ip_rank` |
| `review_scripts/poi_per_cycle.py:38` | reads `versions` | unchanged | reads `ip_rank` (old saves keep `versions`) |
| `review_scripts/cmp_fig_full.py` / `cmp_save.py` | the save's CSVs | the 3 new CSVs: `cmp_fig_full` prints them NEW and exits 1 (expected — exactly those 3 lines); `cmp_save` never lists current-only files | `*_poi_zones.csv` (header change) diffed keyed by column, not positionally |
| Docs naming the save's CSV set (`/compare` + `/commit-save` skills, WORKFLOWS:191, `review_scripts/README.md`) | 24 CSVs | 27 (PR5) | — |
| ARCHITECTURE.md:294 | cond5 = V30 | — | the touch |

### 8.3 Landing A — the selector, exported, nothing switched

1. The helpers of §8.1 (agreed direction + the two qualified copies, the per-frame stamp, `_big_flags`,
   `iter_unfilled_imbalances`, `fib_zone_bounds` + `ic_touches_zone`, `find_ic_candidates(ic_direction=)`, the cap
   table, `POIConfig()` in the orchestrator, `_poi_fib_records`).
2. `patterns/institutional_pattern.py` + `select_ip_ic`.
3. `_run_downstream_pipeline` runs `select_ip_ic(classify_misses=True)` on every gated record; the records go to
   `downstream["ip_candidates"]`, the main's attrs / meta and `LowerTFResult.ip_candidates` (shifted by the mirror).
4. `*_ip_candidates.csv` per lens — one row per (record, candidate) + one `IP: none` row per record without a
   pattern; explicit columns (an empty list writes the header): `structure_path_id`, `sub_id`, `structure_id`,
   `cycle_id`, `bos_anchor_idx`, `cts_anchor_idx`, `sd`, `ic_idx`, `candle_type`, `agreed_dir`, `touches_zone`,
   `ip_rank`, `reason`, `pattern_anchor_idx`, `opp_leg_first_idx`, `pattern_end_idx`, `pattern_confirmed_idx`,
   `ip_leg_candle_counts`, the three stacks' type / ratio / body pips, `picked`, `ip_tiebreak`,
   `ip_tb_runner_up_ic_idx`, `ip_tied_count`, `ip_tb_imbalance_first_c2_idx`.
5. Docs in A's commit: the save's CSV set 24 → 27 (the `/compare` + `/commit-save` skill tables, WORKFLOWS,
   `review_scripts/README.md`, memory counts + a save-format boundary); POI_ZONES_SPEC (the IP-candidates CSV, "not
   yet drawn"); IMBALANCE_FILL_SEMANTICS (`iter_unfilled_imbalances`); GLOSSARY (agreed direction, true doji,
   volume-only doji, split candle — the CSV shows `agreed_dir`; cross-link "qualified candle") (PR5).

**`/compare` prediction (A):** the 24 CSVs md5-equal, the full figure JSON equal, hovers equal (`cmp_fig_full` +
`cmp_save --names-from` + `cmp_hover`); current-only = exactly `…_ip_candidates.csv`,
`…_M15_confluence_ip_candidates.csv`, `…_M15_counter_ip_candidates.csv` (`cmp_fig_full` exit 1 on exactly those 3
NEW lines); no new WARNING lines in run.log; tests 1107 + the new ones, all pass; replay time = today + the IP pass
(measured, stated). **A's check (the charts do not change):** each CSV vs the round-6 prototype per record — H1 3
records (607 3A.iii / 732 3A.iv / 832 3C.ii); confluence 20, counter 7 (the prototype runs 21 / 8: sub 3 cyc 3 is
gated out — active, not locked, ended); every difference explained (the prototype takes the zone from candle extremes,
the code from `FibState` prices — an exact-edge touch can differ). Then the census (§8.4) and `/commit-save` → S_A.

### 8.3a Landing A — as built (2026-10-06)

- **Code:** new `patterns/institutional_pattern.py` (`leg_variants` → the stacks in one classifier call →
  `rank_variant`; `classify_ics` / `classify_ip`; `IPResult`; the rank table `SINGLE_OPP_RANK` / `MULTI_OPP_RANK`);
  `candles_v2.agreed_direction` + `AGREED_SPLIT` + `big_ratio_flags` (factored out of `classify_big_flags`);
  `candle_classifier.candle_params_for_timeframe` + the `attrs["timeframe"]` stamp; `imbalance.iter_unfilled_imbalances`;
  `poi_zones`: `POIConfig.ip_leg_cap_by_tf`, `find_ic_candidates(ic_direction=)`, `fib_zone_bounds`, `ic_touches_zone`,
  `IPCandidate` / `IPSelection` / `select_ip_ic`, `_poi_record_gate`, `IPCandidateRecord` / `derive_ip_candidates`;
  the orchestrator (`POIConfig()`, step 8b, `downstream` / `meta` / attrs `ip_candidates`); `LowerTFResult.ip_candidates`
  + the mirror (`_IP_CANDIDATE_IDX_FIELDS`); `export_zones.export_ip_candidates` + the 3 `run_replay` calls; the two
  qualified copies (`wave_candles._is_qualified`, `wvmi._find_temporary_lp`) read `agreed_direction`.
- **Deviations from §8.1 / §8.3 (my calls, recorded here):** (1) the frame stamps ONLY `attrs["timeframe"]` (a str);
  the IP derives its params with `candle_params_for_timeframe` — one source, and a params object in attrs would be
  deep-copied on every pandas `__finalize__` from the classification on. (2) A touch-missing candidate keeps its own
  pattern / reason in the CSV (`touches_zone` False, never picked) instead of reason `no_touch` — the T1 review sees
  what it would have ranked. (3) `legs` is not stored: the CSV locates it from `ic_idx` + `ip_leg_candle_counts` (the
  legs are contiguous) — a stored index string was left slice-local by the mirror on the first replay (caught by the
  prototype check, fixed before the formal replay); a failing candidate's counts / stacks are its first failing
  variant's. (4) The none row's reason is `no_pattern` (candidates, none picked) or `no_candidates`.
- **Tests:** `tests/test_institutional_pattern.py` (89) + 2 in `test_event_meta_idx_keys.py` (the new module has no
  index-like dict keys; `render_sub_projection` mirrors the candidates to both lenses) → **1198** (1107 + 91; `pytest`
  from the repo root incl. the OANDA smoke). **Mutation:** `plan_h_inputs/mutants_plan_h_a.py`, 66 mutants (every IP
  rule branch, the selection / TB1 / TB2a, the gate + rows, the mirror, the wiring) — first run 62/66 KILLED; the 4
  survivors (a true doji INSIDE the initial run, the TB1 runner-up type order, the orchestrator's `POIConfig()`, the
  ratio rounding) got a test each → **66/66 KILLED**.
- **`/compare` (vs the Week 8 save, named explicitly):** `cmp_fig_full` 24/24 md5-equal, the 3 charts' full figure
  JSON equal, NEW = exactly the 3 `*_ip_candidates.csv`; `cmp_save --names-from` 0 real cells, figures equal;
  `cmp_hover` 0 / 0 / 0 — **== the prediction**. Replay 70.8 s (multi_tf_dual 10.0 s; the window's 41–96 s band); the
  fetch gate 4228 M15 candles; run.log's 2 WARNING lines are `parent_tables`' degenerate (1,0) / (1,1) (H1 events
  unchanged → pre-existing).
- **A's check vs the round-6 prototype:** H1 3 / 3 records equal (607 3A.iii, 732 3A.iv, 832 3C.ii); confluence 20 /
  21 and counter 7 / 8 equal — the one difference on each is sub 3 cyc 3 `[3304, 3611]`, which the record gate excludes
  (active, not locked, ended) exactly as `derive_poi_zones` does (§8.8 PR11). TB2 as measured: 1875 over 1842 (reference
  c2 1892), 4083 over 4136 (4132). `pattern_confirmed_idx` on the R1 rows: 2383 → 2386, 2704 → 2706, 3388 → 3391.
  Rows: H1 78 (40 touch-misses), confluence 2985 (2530; 6 none rows), counter 409 (221; 0 none rows).
- **Landing review (2026-10-06; 2 read-only lenses, 453,157 tokens — est. ~0.5M):** (1) code conformance (233,866):
  0 BLOCKER — every §1 / §6 rule as decided, the 4 deviations implementation-only; a 24,106-call `select_ip_ic` sweep
  over BOS × as-of CTS (H1 + M15, both sd): 0 exceptions, the leg / touch / `pattern_confirmed_idx` invariants on ~42,800
  patterns; no look-ahead in-flight (6,207 calls on the frame truncated at the as-of CTS: identical selections); B's
  three frames carry the stamp, the 0-based index and the columns. Two B items (→ §8.5): a SECOND swallowing scope
  above the resolver for sub geometries, and the cost (+7.8–11.2 ms per resolver call, ~550 calls ≈ +4–5 s). Nits
  folded: `_check_index` = `df.index.equals(pd.RangeIndex(len(df)))`; `attrs["pair"]` read directly (no silent
  non-JPY default); `derive_ip_candidates` asserts like `derive_poi_zones` on a locked fib without CTS_ESTABLISHED;
  `ip_tied_count` documented (the group the DECIDING rule broke: TB1 the rank tie, TB2 the TB1 survivors); the main
  frame's fixed "H1" stamp → register §A. (2) Test gaps (219,291): 30 new mutants — 7 killed, 22 real survivors (no
  sd = −1 pattern test: the stacks' max-high / min-low, the sell zone; `select_ip_ic` never ran on an M15 frame; the
  CSV columns; the mirror with several subs per lens; TB1 / TB2 internals; R1's run == cap boundary; …), 1 equivalent
  (N18) → its 16 killing tests folded + 2 of mine (R1 past the frame; a mirrored record's legs) + 3 for the fail-loud
  nits. **Tests 1219; `mutants_plan_h_a.py` 98/98 KILLED** (66 + the review's 29 + 3). Fixture sizes sit near H1's
  2.2-pip floor (a floor recalibration — item 4 — makes them fail loudly, not pass vacuously).
- **Saved (2026-10-06):** source `e1c597c`, save `20261006_120513_e1c597c` = S_A (save commit `6a3a81d`, trunk
  `492e7a8`; reuse mode of the verified confirming replay — FETCH GATE PASS, the saved 27 CSVs + 3 figures md5 / JSON
  equal to it). **Next:** §8.4.

### 8.4 Between A and B — the variant replays that write B's prediction

- **Three variants** through scratch trees (`review_scripts/replay_variant.py`: a `git archive` of A + a minimal
  behaviour patch, no meta migration), each diffed against S_A: **V1** the drawn POI only (`derive_poi_zones` from the
  picks + the touch re-check + the `pattern_confirmed_idx` floor; the resolver unchanged), **V2** the MS resolver only,
  **V3** both. V1 / V2 attribute every delta to one cause (one cause per `/compare`, inside L1's single landing B);
  B's `/compare` must equal V3 + the format-only deltas (PR2).
- **The resolver log** (V2 / V3; behaviour-neutral, wraps `structure_engine.compute_poi_inners_for_cycle` — the only
  name MS reads it by, resolved at call time): per call the caller (main / sub geometry / probe Phase 2), sid, cycle,
  the as-of anchors, today's inners vs the IP inner, the final drawn pick (PR12), the IP time (PR8).
- **The hand cross-check:** H1 POIs 5 → 3 rows (607 / 732 / 832), triggers 710 → 748 and 926 → 939 (§3 round 5) must
  appear in V3, or the difference is explained before B.
- **The test census** (a pytest plugin, `census_plugin.py`'s recipe) on A's suite with the V3 patch: per test the
  POIs built, the resolver's inners, the IP pick or none → every moved pin listed as re-pin / re-derive / retire (PR7).

### 8.4a Between A and B — as measured (2026-10-06)

**Tools** (`plan_h_inputs/`): `variant_patch.py TREE V1|V2|V3` (the minimal behaviour patches on a `git archive` of
A — V1 7 edits / V2 1; no meta migration: `versions` = `[ip_rank]`, activation entries carry `["touch"]`);
`replay_variant_h.py TREE A_POI_ZONES LOG.json [--nolog]` (the real replay through the tree, cwd = a short work dir
with `oanda.cfg` — `%TEMP%/ph/<v>`: under the session scratchpad the M15 CSV names exceed MAX_PATH; logs every
resolver call (caller from the stack, as-of anchors, A's inners for the same arguments, both times, the IP pick),
every other `select_ip_ic` call (drawn / CSV), `select_ip_ic` errors before they propagate, and per drawn POI its
history with cond5 forced (A's sweep at a 0 % threshold) and without the R1 floor); `variant_report.py BASE CUR`
(keyed: md5, POIs per lens keyed by the sub's (direction, `starting_idx`), events, triggers, subs, `_final.csv`);
`resolver_log_report.py LOG`; `census_h_plugin.py` + `census_h_diff.py` (per test: POIs, direct activation histories,
resolver answers, picks, outcome — A's tree vs V3's).

**V1 (the drawn POI only) vs S_A:** H1 POIs 5 → 3 (607 3A.iii / 732 3A.iv / 832 3C.ii); H1 events unchanged; the
zone-proximity scan reads the new POIs → counter triggers 2843 → 2995 (H1 710 → 748), 3707 → 3759 (926 → 939: the
scan re-reads the drawn IC 832 from 926 on), the follower `subsequent_counter` 4083 → 4159 (H1 1020 → 1039); 14 CSVs
differ. **V2 (the resolver only):** H1 `CTS_CONFIRMED` (1,2) 926 → 939 (`sd_zone_proximity`, `trigger_inner` 0.5772
IC 865 → 0.57866 IC 832 — the hand cross-check's "≥ 939": exactly 939) → the counter `first_counter` 3707 → 3759
(the scan starts at CTS_CONFIRMED); conf sub 1 cyc 1's `CTS_CONFIRMED` 1902 survives (`trigger_inner` 0.5809 IC 1889 →
0.58074 IC 1875); no M15 event moves; 12 CSVs. **V3 (both) = the union** on every keyed cell: 19 CSVs differ, the 8
equal = `_raw`, `imbalance_instances`, `structure_levels`, the 3 `*_ip_candidates.csv`, `M15_unresolved_triggers`,
`M15_counter_structure_events`. The hand cross-check holds: POIs 5 → 3, triggers 710 → 748 and 926 → 939. **D2
holds on the final fib:** V3's drawn POIs == A's picks on every record (H1 3 / 3, confluence 14 / 14 + 6 no-pattern
records, counter 7 / 7).

**The plan's predicted list vs V3:** WVMI (1,2) `fp_idx` + momentums do NOT move (the FP search to CTS_CONFIRMED + 10
finds the same FP; only `triggered_by_event_idx` / `trigger_inner` move); NEW: conf + counter sub 7 cyc 0
`cycle_collapsed` False → True (sub 7 now starts at 4159 = its cycle 1's start → cycle 0 collapsed: its fib record
`start_idx` → empty / `inactive`, its KL zone ends at 4159, its POI 4035 never active).

**The activation attribution (V3, 18 drawn POIs):** the 4 never-active POIs are all COLLAPSED cycles (lifecycle
floor > scan end — not the IP): H1 732 (as in A), conf 1875 (A's 1889 / 1890 also never active), 2704 (A's 2702 /
2706 too), 4035 (sub 7's cycle 0, above). **R1's floor binds nowhere** on the window (history == without it). **The
as-of touch (T1) delays 2:** sub 3 cyc 2 IC 2837 2992 → 3047 and sub 7 cyc 2 IC 4083 4179 → 4222 (cond1 + cond3 hold
from the first date; the IC touches only a later as-of zone) — chart-review items. No POI deactivates on cond5.

**The resolver log (V3, 450 calls, 0 `select_ip_ic` errors, no call failed inside the kept `try`):**

| caller | calls | IP = A | IP ∈ A | IP ∉ A | IP none, A some | IP some, A none | both empty | ms / call IP vs A |
|---|---|---|---|---|---|---|---|---|
| main (H1) | 58 | 1 | 2 | 14 | 36 | 0 | 5 | 5.5 vs 2.1 |
| sub geometry | 313 | 35 | 21 | 182 | 42 | 2 | 31 | 6.3 vs 2.3 |
| probe Phase 2 | 79 | 0 | 3 | 48 | 22 | 0 | 6 | 9.4 vs 5.2 |

PR12: the as-of picks differ from the final drawn pick in 185 of 450 calls (main sid 1 cyc 1: 4 of 7; most on subs,
where MS reads the INTRA fib and the drawn POI the cross fib — D5, item 2); 146 calls have no drawn record on their
frame (probe Phase 2). Only one MS outcome moves (H1 926 → 939). **Cost (PR8):** +1.8 s of resolver time per replay
(+3.4 / +4.0 / +4.2 ms per call; under load, 3 replays in parallel) — below the review's +4–5 s; no memoization.
Wall clock is not separable: 4 sequential runs A 87.0 / 61.6 s, V3 78.0 / 38.7 s (machine noise).

**The census (A's suite, A's tree vs V3's; 1219 tests):** 18 FAIL, 99 MOVED-SILENT (pass, but their POIs / resolver
answers / direct histories moved), 174 touch the POI layer unmoved, 927 never touch it.

| Group | Tests | B action |
|---|---|---|
| F-a `KeyError: 'timeframe'` — a hand-built frame calls the resolver directly (PR1 as designed) | `test_e3a_mutation_pins` `test_compute_poi_inners_uses_its_fill_horizon`, `test_e3ap_compute_poi_inners_uses_its_snapshot_horizon`; `test_imbalance_c3_knowability` `test_guard_ms_inflight_select_call_is_uncut`, `test_m1_ms_inflight_keeps_the_inner_fibtracker_creates_no_fib` | re-fixture: stamp the frames (the 3 spies then pass unchanged); M1 (`inners == [1.13]` on a lone candle) → a pattern around its IC, re-derived |
| F-b the MS outcome moves | `test_ms_reversal_terminal` P3 `test_a_reversal_inside_a_post_apply_range_backfill_does_not_finalize_the_range[±1]` (`RANGE_STARTED` 11 → 12: the fixture's POI inner is gone, only the BOS inner is left) | re-derive the pin (12) and re-check it still kills the fix's revert — or re-fixture so the IC has a pattern |
| F-c the fixture's IC has no pattern (PR7) | `test_poi_activation_moment` ×10 + `test_imbalance_c3_knowability::test_multicycle_poi_full_history` (`_make_second_cts_moment_after_anchor_data` IC 7, `_make_multicycle_data` IC 12) | a new fixture with anchor ≠ moment AND a pattern; pins re-derived |
| F-d ORDER | `test_poi_activation_moment::test_a_locked_fib_without_cts_established_fails_loudly` (V3 selects before the no-CTS_ESTABLISHED assert → no pick → `continue`) | B: ONE record gate for `derive_poi_zones` + `derive_ip_candidates` — the lifecycle gate, then the CTS_ESTABLISHED skip / assert, THEN the selection (also skips the IP on skipped records) |
| S-1 POIs vanish (A 1–2 → 0), the test still passes — its POI part goes vacuous | `test_e4_simulation` ×9 (`test_downstream_outputs_do_not_depend_on_the_idx_role` ×7, `…_pattern_update_idx_role` ×2), `test_institutional_pattern` ×2 (`test_downstream_pipeline_*`), `test_orchestrator_reversal_source` ×1, `test_plan_e_role_pins::test_lagging_est_fib_lifecycle_is_timed_at_the_moment[h1]`, `test_poi_activation_moment` ×2 (`…raises_on_cts_established_without_confirmed_at`, `test_poi_lookup_reads_confirmed_at_without_fallback`) | the new fixture where POIs are the point; a non-empty assert before every POI equality |
| S-2 POIs kept, values moved | `test_event_meta_idx_keys` ×13, `test_render_sub_projection` ×6, `test_plan_g_wvmi` ×4, `test_pooled_structure_build` ×2 (`_poi_sig` reads `versions`), `test_meta_key_renames` ×2 | the new keys in `must_see`; `_poi_sig` + `ip_rank` |
| S-3 resolver answers moved, MS outcomes not | 59 MS / probe / chart tests (`test_ms_bounded_equals_truncated`, `test_ms_stop_after_cts`, `test_unified_probe`, `test_e4_simulation` emitters, …) | none (they compare runs or pin arguments); B adds the resolver pins (V ≠ IP fixture, a non-V pick, the unstamped frame) |
| S-4 direct `_compute_poi_activation_history` calls (`versions` / reason moved) | 8 (`test_plan_e_role_pins` ×3, `test_poi_activation_moment` ×3, `test_imbalance_c3_knowability` ×2) | the new signature (no `variant_thresholds`); the explicit `versions` pins (`test_poi_activation_moment` :110 / :268, `test_zone_proximity` :127 fixture meta) re-pinned |

**B's `/compare` prediction (written 2026-10-06, before B is built) = V3 + the format deltas.** Kept for the check:
V3's outputs in `%TEMP%/ph/V3/artifacts/` (deleted 2026-10-06 after B's check; rebuild with `variant_patch.py`).
1. **B vs V3:** the 24 CSVs other than the 3 `*_poi_zones.csv` md5-EQUAL; every figure attribute equal except the
   POI hover traces' `hovertemplate` + `customdata` (H1 3 POIs × 2 lines; per M15 chart the sub POIs + the H1
   overlay POIs × 2) — trace / shape counts equal (H1 81 / 235, counter 129 / 90, confluence 258 / 178).
2. **The format deltas** (the `*_poi_zones.csv`, keyed by column): `versions` → `ip_rank` (the same position), values =
   the pick's rank (H1 `3A.iii`, `3A.iv`, `3C.ii`); meta: `versions` / `current_versions` replaced IN PLACE by
   `ip_rank`, `ip_leg_candle_counts`, `pattern_anchor_idx`, `opp_leg_first_idx`, `pattern_end_idx`,
   `pattern_confirmed_idx`, `ip_tiebreak`, `ip_tb_runner_up_ic_idx`, `ip_tied_count`, `ip_tb_imbalance_first_c2_idx` ==
   S_A's picked `*_ip_candidates.csv` rows (sub indices entity-absolute — the mirror shifts the 6 index keys; TB2 on
   conf sub 1 cyc 1 1875 over 1842 / c2 1892 and sub 7 cyc 2 4083 over 4136 / c2 4132); activation entries
   `{idx, active, reason}` (no `versions`); every other cell == V3's. Hover: `versions=…` → `IP <rank> · legs a-b-c
   (initial | opp | continuation)` (+ the TB line on the two TB2 POIs); the H1 fib hovers unchanged (no H1 record
   without a pattern).
3. **Hence B vs S_A:** the 8 CSVs above md5-equal; `*_ip_candidates.csv` ×3 unchanged; everything in §8.4a's V3 rows.
4. Tests: 1219 − the census's retired + B's new, all pass; run.log: the same 2 `parent_tables` WARNING lines; time:
   +≈ 2 s of resolver time (not separable from the wall-clock noise).

### 8.5 Landing B — the switch + the rule-3 migration (one commit)

1. `derive_poi_zones` builds the POIs from the picks. Meta (D6 / HV): `ip_rank`, `ip_leg_candle_counts`,
   `pattern_anchor_idx`, `opp_leg_first_idx`, `pattern_end_idx`, `pattern_confirmed_idx`, `ip_tiebreak` (None / "TB1"
   / "TB2"), `ip_tb_runner_up_ic_idx`, `ip_tied_count`, `ip_tb_imbalance_first_c2_idx` — flat ints / None / str (the
   idx guard's rule); `versions` / `current_versions` removed; no alias.
2. Activation: cond5 → `ic_touches_zone` on the as-of zone, reason `ic_not_touching_fib_zone` (R3); `first_active =
   max(CTS-established moment, ic_idx, lifecycle floor, pattern_confirmed_idx)` (R1).
3. The MS resolver → `select_ip_ic` outside the `try` (PR1) — AND the second swallowing scope above it for sub
   geometries: `entity_df_mutation.py` ~:1135–1147 wraps `compute_bounded_structure` in `except (ValueError,
   IndexError): return None`, so an IP ValueError / IndexError inside a sub's MS would silently drop the WHOLE sub
   (landing review). Narrow it to its documented failure (or the IP raises only KeyError / AssertionError); pin: an IP
   error inside a sub geometry propagates, and the resolver on an unstamped `df_probe` (the Phase-2 frame) raises.
   Name both scopes in LANDMINES. Cost: +7.8–11.2 ms per resolver call (75 % the stacks' one `classify_candles` call;
   ~550 calls ≈ +4–5 s on the 70.8 s replay) — V3's resolver log confirms; memoize the stack types per
   (frame, first, last) only if needed.
4. The mirror shift list + `test_event_meta_idx_keys` (§8.2 row).
5. Exports (`ip_rank` …), the hover builder × 3 sites, the H1 fib hover, the debug dump, `poi_per_cycle.py`.
6. Docs in the same commit: POI_ZONES_SPEC §3 rewritten as the IP (from §1 here) + "Zone Data Fields" + §4
   activation; ARCHITECTURE.md:294; CHARTING_SPEC (hover); GLOSSARY (institutional pattern, the three legs, `ip_rank`,
   `pattern_confirmed_idx`); LANDMINES (one selector for MS + drawn; the resolver's fail-loud scope); the charts'
   `_poi_confirmed_idx` comments; register G "Cascade-viz hover extras" closed; the save-format boundary
   (`*_poi_zones.csv`: `versions` → `ip_rank` + the IP columns) in memory `project_plan_e_event_convention.md`.

**`/compare` prediction (B) = V3 + the format deltas**, written before the replay, at least: H1 POIs 5 → 3 rows (607
/ 732 / 832); H1 sid 1 cyc 2 `CTS_CONFIRMED` 926 → V3's value, and everything keyed on it — the CTS KL zone (1,2)
`confirmed_idx` / `confirmation_method`, fib (1,2) `locked_at`, WVMI (1,2) `fp_idx` + both momentums (FP searched to
CTS_CONFIRMED + 10), the `_final.csv` MS columns from 926, the trigger scan start; the H1 triggers 710 → 748 and 926 →
939 (+ the followers); the H1 WVMI gate attributions and the confluence WVMI `triggered_by_*` cells; M15 confluence
subs 0–3 POIs == A's picks (19 → 10 rows; 1902 survives); the counter subs and confluence subs 4 / 6 / 7 re-derived,
keyed on (direction, `starting_idx`) since `sub_id` is creation order; the `*_poi_zones.csv` columns; the hovers;
the replay time (from V3; the landing review measured ≈ +4–5 s). Then the chart review (the no-POI cycles, the touch rule, the re-derived counter lens).

### 8.5a Landing B — as built (2026-10-06)

- **Code:** `poi_zones`: `IPRecordSelection` + `select_ip_records` (the lifecycle gate → the cycle's CTS_ESTABLISHED
  skip / locked-fib assert → `select_ip_ic(classify_misses=True)`, computed ONCE per pipeline run; the orchestrator
  hands the same list to `derive_poi_zones` and `derive_ip_candidates` via `ip_records=`); `derive_poi_zones` draws the
  pick (0–1 POI per record) with the IP meta in place of `versions` / `current_versions`; `_compute_poi_activation_history`
  cond5 = `ic_touches_zone` on the as-of `fib_zone_bounds`, reason `ic_not_touching_fib_zone`, `first_active` +
  `pattern_confirmed_idx`; the resolver runs `select_ip_ic` after its `try` and re-raises any IP error as
  `IPSelectionError`; `select_ic_variants` marked UNWIRED (X1). `entity_df_mutation._ZONE_META_IDX_KEYS` + the 6 IP
  index keys; the geometry scope's comment. `export_poi_zones` `versions` → `ip_rank`. `_zone_render.poi_ip_hover_text`
  (HV) at the 3 POI hover sites; the H1 fib hover's `IP: none (<reason>)`. `poi_per_cycle.py` reads both formats.
- **Deviations from §8.5 (my calls):** (1) the sub-geometry scope (step 3): neither "narrow it to its documented
  failure" nor "the IP raises only KeyError / AssertionError" — the resolver re-raises ANY `select_ip_ic` error as
  `IPSelectionError` (a RuntimeError, chained), so no ValueError / IndexError / LookupError scope above it matches;
  the geometry scope stays as documented (`geometry_failed` never occurs on the window). (Corrected §8.5c C1: "today
  or later" overclaimed — an `except RuntimeError` / `Exception` WOULD swallow it; none encloses an IP path today,
  and `tests/test_exception_scopes.py` now fails on any new one — LR2.) (2) ONE record list (`select_ip_records`)
  for the drawn POIs and the CSV, with the CTS_ESTABLISHED assert BEFORE the selection (the census F-d: V3 selected
  first). (3) The activation takes `config: POIConfig` in place of `fill_threshold` (unused in the sweep) /
  `fib_min_pct` / `fib_max_pct` / `variant_thresholds`; `pattern_confirmed_idx` is required. (4) "IP: none" is appended
  to the fib hover's status cell (template unchanged — no H1 fib hover moves on the window: every H1 record has a pick).
- **Tests (the §8.4a census, actioned):** F-a the hand-built frames go through the real classifier (`_ip_ready`); M1
  gets a real pattern around IC 32 (`2A.i`, `inners == [1.13]` kept). F-b P3's range 11 → 12 (re-derived; the pin
  still kills the reverted back-fill guard, both sd). F-c new fixtures `test_unified_probe._make_ip_moment_after_anchor_data`
  (CTS_ESTABLISHED (0,1) anchor 12 / moment 13; IC 9 `1A.iii`, `pattern_confirmed_idx` 12 — the R1 floor sits before
  the moment, so it never masks the moment rule) and `_make_ip_multicycle_data` (cycle 2 anchor == moment 17; IC 14
  `1A.iii`; the same Plan F fill-at-scan-end case shifted by 2) — `test_poi_activation_moment` re-pinned on them, the
  c3 full-history pin re-pointed. F-d passes by (2). S-1 `test_e4_simulation` + 5 IP params × 2 modes with a non-empty
  POI assert (`_IP_SECOND` in cross_cycle draws none: its cross fib's zone lies far below the IC — stated in the test);
  the pattern-update variant stays POI-empty (its cycle 3 has no pattern; the reader is pinned at sweep level). S-2 the
  idx guard's `must_see` + `ip_tied_count`, `_poi_sig` + the IP fields, the zone-proximity fixture meta. S-4 the 6
  direct sweep calls on the new signature. New `tests/test_plan_h_landing_b.py` (27): the as-of touch both ways + its
  reason, both edges inclusive, the R1 floor (4 cases incl. past the frame), the resolver where V (36) and IP (32)
  disagree, no pattern → `[]`, the unstamped frame / a re-raised ValueError / an IP error inside `_build_geometry` / an
  unstamped MS run, D2 on the real collaborator (both fixtures + the sd = −1 mirror), the meta key order, the tiebreak
  copy, no POI without a pattern, the hover builder (4) + the H1 chart + the H1 overlay sites, the exporter columns,
  the R1 floor through `derive_poi_zones` (an injected pick confirmed at 14), and a SOURCE pin of the M15 sub-POI hover
  call (no test renders a lens chart with sub POIs — a gap for the landing review; the replay checks the text).
  **Tests 1258** (1219 + 39; `pytest` from the repo root incl. the OANDA smoke), all pass.
- **`/compare` vs S_A == §8.4a's prediction:** FETCH GATE PASS; run.log = the 2 `parent_tables` WARNING lines + the 4
  UNRESOLVED `degenerate_parent_cycle` rows (unchanged); vs V3: 24 / 27 CSVs md5-equal, the 3 `*_poi_zones.csv` ==
  V3 + exactly the format delta (`cmp_poi_b.py`, IP values == S_A's picked rows, 3 TB2 POIs), figures equal but 44
  POI hover traces (`customdata` + `hovertemplate` only: H1 4, M15 sub POIs 32, H1 overlays 8); trace / shape counts
  H1 81 / 235, counter 129 / 90, confluence 258 / 178; the keyed B-vs-S_A report == V3's line for line. Replay 81.4 s
  wall (machine noise; §8.4a: ≈ +2 s of resolver time).
- **Mutation:** `plan_h_inputs/mutants_plan_h_b.py` (32: every rule + call site of B) — first run 29 / 32 KILLED; the
  3 survivors (B08 the call site's R1 term, B11 the meta's `pattern_confirmed_idx` — R1 never binds on the fixture —,
  B31 the M15 sub-POI hover call) got a test each → **32 / 32 KILLED**.
- **Chart review (2026-10-06):** the user reviewed B's charts and chose to **debug them in a NEW session**: questions
  about the changes, chart elements updated to make the differences reviewable, and rule experiments — each output
  compared against S_B (this landing's save) AND S_A (the pre-IP iteration). The points raised: §8.5b.
- **The landing review — DEFERRED (the user, 2026-10-06):** the debugging session will change B's own rules, so ONE
  review (2 read-only lenses, ~0.55M: code vs plan — the resolver's fail-loud scope, the mirror shift, the hovers —
  and test gaps, incl. the M15 sub-POI hover pinned only by a source check) runs after those rules settle, over B +
  the changes. Saving first is safe: B == V3 (an independent route), 32 / 32 mutants, and no reader of `versions` /
  `current_versions` is left (grep, 2026-10-06: only the review tools that read both formats — and, §8.5c C9, two
  older measurement tools, `plan_f_inputs/shadow_measure.py` and `plan_h_inputs/census_h_plugin.py`, whose
  `.get("versions", [])` reads an empty list on post-B saves; kept as history). **Ran 2026-10-06c: §8.5c.**
- **Committed + saved (2026-10-06):** source `e54f509`, save `20261006_151106_e54f509` = **S_B** (save commit `b0556fc`,
  trunk `0d96c12`; reuse mode of the verified replay, FETCH GATE PASS; 27 CSVs) = this branch's `/compare` baseline
  (`LATEST_week9-entries-decisions`). S_A stays the pre-IP reference: name it explicitly (`cmp_save.py` positional).

### 8.5b Landing B — the chart-review reference points (2026-10-06)

**Re-derived on S_S6 (2026-10-09): §8.9 R0** — what moved after the cross-cycle fib revamp; the tables below are
B's record. The points raised at B's chart review, kept as the reference for the debugging session (the user, 2026-10-06:
"document … so I can reference those points later in our debugging session"). Re-derived from B's replay outputs and
S_A's CSVs — not copied from the chat. **Conventions:** indices entity-absolute; times UTC as the charts show them
(M15 `i` = H1 candle `i // 4` + 15 min × `i % 4`; both frames start 2025-11-16 22:00); "A drew" = S_A's POIs, the
V30 / V60 / V90 picks the pre-IP charts show (S_A's charts: `artifacts/commits/week9-entries-decisions/
20261006_120513_e1c597c/*.html`; B's = S_B's: `…/20261006_151106_e54f509/*.html`); a sub is named by `sub_id` (creation order) with its stable key (direction,
`starting_idx`); "both lenses" = the unique sub appears in the counter AND the confluence chart.

**1. Cycles that lost their POI (C2) — confluence lens only.** Every one is `no_pattern` in B's
`*_M15_confluence_ip_candidates.csv` (candidates exist; none forms a pattern whose IC touches the zone). Their fib
is drawn as before; the M15 fib hover does NOT say why there is no POI (only the H1 fib hover carries `IP: none
(<reason>)`, and no H1 record lacks a pick) — a candidate chart aid. **Corrected 2026-10-06 (§8.9 F1):** the M15
charts drew NO fib at all (the renderer was H1-only); the M15 fib layer with that hover landed as the first chart
aid. Sub 0 cyc 0 and sub 1 cyc 0 are COLLAPSED cycles (ended before their sub went live) — hidden, like their KL /
POI (F1b).

| Sub (dir, starting) | Cycle · fib BOS → CTS anchor | A drew (A's activation) | Why A's IC is no pattern | Candidates · touching · patterns (none touching) |
|---|---|---|---|---|
| sub 0 (buy, 454) | 0 · 454 (11-21 15:30) → 784 (11-27 02:00) | 678 (11-25 23:30) V30/V60/V90 (never active) | 678 maru: `continuation_not_big_maru` (legs 676 · 677–678 · 679–681) | 146 · 13 · 8 — best off-zone 778 / 782 1A.iv (11-27 00:30 / 01:30) |
| sub 0 | 1 · 454 → 1020 (12-01 13:00) | 678 V30/V60/V90 (active from 1020) | the same | 261 · 10 · 18 — 778 / 782 1A.iv, 996 1B.iii |
| sub 0 | 2 · 454 → 1245 (12-03 21:15) | 678 V30 (active from 1224) | the same | 362 · 4 · 25 — 778 / 782 / 1117 1A.iv |
| sub 1 (sell, 1797) | 0 · 1797 (12-11 15:15) → 1816 (12-11 20:00) | 1810 (12-11 18:30) V30 + 1808 (12-11 18:00) V60/V90 (never active) | 1810 pinbar (1809 · 1810 · 1811–1812) and 1808 normal (1805–1807 · 1808 · 1809): both `continuation_not_big_maru` | 8 · 7 · 1 — 1813 2B.iv (12-11 19:15) |
| sub 6 (sell, 3760) | 0 · 3760 (01-13 02:00) → 4000 (01-15 14:00) | 3809 (01-13 14:15) V30/V60 + 3808 (01-13 14:00) V90 (active from 3819) | 3809 maru: `initial_fails_multi_opp` (3807 · 3808–3809 · 3810–3812); 3808 pinbar: `ic_not_last_opp` (3809 follows in its opp leg) | 109 · 9 · 10 — 3899 1A.i (01-14 12:45), 3815 1A.iv (01-13 15:45) |

Also no POI, in A as in B: sub 1 cyc 2 (1934 → 1945; 3 candidates, none touching). Sub 0's later cycles 3 / 4 pick
687 3C.iv (11-26 01:45) — off-zone for cycles 0–2, whose zone is measured to different CTS anchors.

**2. The touch rule (T1).**
- **(a) The as-of touch delays activation** (cond5 = the IC touches the 61.8–80 zone as of each CTS event; §8.4a —
  "without the touch" = A's sweep on the same pick):

  | Sub (both lenses) | Cycle · fib | B's IC | Active without the touch | Active in B | A drew (active from) |
  |---|---|---|---|---|---|
  | sub 3 (sell, 2639) | 2 · 2758 (12-26 15:30) → 3047 (12-31 15:45) | 2837 1A.iv (12-29 11:15) | 2992 (12-31 02:00) | 3047 (12-31 15:45) | 2961 (12-30 18:15), from 3047 |
  | sub 7 (buy, 4027) | 2 · 4027 (01-15 20:45) → 4222 (01-19 21:30) | 4083 1A.iv (01-16 10:45; TB2 over 4136) | 4179 (01-19 10:45) | 4222 (01-19 21:30) | 4127 V90 from 4179, 4136 V60 from 4195, 4145 V30 from 4222 |

- **(b) A pattern whose IC misses the zone while an earlier opp candle touches it** (§3 round 4; decided T1: no
  pattern): 2375 3C.iii (12-19 15:45; conf sub 2 cyc 0 / 1) and 4048 3C.iv (01-16 02:00; sub 7 cyc 2). Both are
  outranked anyway — no pick changes. 4048 is also A's POI for sub 7 cyc 0 (B picks 4035 2C.iii there).
- In general a pattern off the zone is common: the CSV rows with `ip_rank` set and `touches_zone` False (sub 0's
  records alone hold 8–25 each).

**3. The re-derived counter lens (C3)** — the zone-proximity scan reads the new H1 POIs (V1) and the moved H1
`CTS_CONFIRMED` (1,2) (V2); each M15 trigger is the last M15 candle of its H1 trigger hour:

| Counter record → sub | S_A trigger | B trigger | Cause |
|---|---|---|---|
| (0,1) `first_counter` → sub 3 (sell, 2639) | 2843 (12-29 12:45) = H1 710 | 2995 (12-31 02:45) = H1 748 | H1 (0,1)'s nearest inner 611 0.57850 → 607 0.57712 (deeper) |
| (1,2) `first_counter` → sub 5 (buy, 3621) | 3707 (01-12 12:45) = H1 926 | 3759 (01-13 01:45) = H1 939 | the scan starts at H1 (1,2)'s `CTS_CONFIRMED`, 926 → 939 |
| (1,2) `subsequent_counter` → sub 7 (buy, 4027) | 4083 (01-16 10:45) = H1 1020 | 4159 (01-19 05:45) = H1 1039 | the drawn IC 865 → 832 (V1) |

- Sub 3's counter record starts at 2995; the sub itself still starts at 2829 (its confluence `reversal` record).
- Sub 5: 3707 → 3759 … 4083 → 4159 (`same_dir_replacement` by sub 7).
- Sub 7 starts 4083 → 4159 (both lenses), its cycle 1's start → **cycle 0 collapsed** (`cycle_collapsed` True: fib
  record inactive, KL zone ends 4159, POI 4035 2C.iii never active; A's 4048 was active 4083–4104 and from 4119).

**4. H1.**

| Cycle (fib) | A drew | B (IP) | Activation |
|---|---|---|---|
| sid 0 cyc 1 buy (591 → 652) | 611 V30 (12-22 09:00), 607 V60/V90 | 607 3A.iii (12-22 05:00): 606 · 607 · 608–610 | 652, unchanged |
| sid 1 cyc 1 sell (689 → 761) | 732 | 732 3A.iv (12-30 10:00): 729–731 · 732 · 733–735 | never — a collapsed cycle, hidden as in A |
| sid 1 cyc 2 sell (826 → 905) | 865 V30/V60 (01-07 23:00), 860 V90 (01-07 18:00) | 832 3C.ii (01-06 14:00): 828–830 · 831–832 · 833 | from 902 (off 952, on 954, off 992, on 998) — A's 860 history |

H1 `CTS_CONFIRMED` (1,2) 926 (01-12 12:00; `sd_zone_proximity`, IC 865's inner 0.5772) → **939** (01-13 01:00;
IC 832's inner 0.57866), and everything keyed on it (§8.4a); the H1 triggers 710 (12-29 12:00) → 748 (12-31 02:00)
and 926 → 939 (01-13 01:00).

**5. The new hovers (HV).** H1 `IP 3A.iii · legs 1-1-3 (606 | 607 | 608–610)`, `IP 3C.ii · legs 3-2-1 (828–830 |
831–832 | 833)`; M15 sub 7 cyc 2 `IP 1A.iv · legs 2-1-2 (4081–4082 | 4083 | 4084–4085)<br>TB2 over 4136 (reference
imbalance 4132)`; conf sub 1 cyc 1 1875 3C.iv TB2 over 1842 (c2 1892).

### 8.5c Landing review — B + F1 + F1·1 (2026-10-06c; §6 LR1–LR2)

Two read-only Agent-tool lenses in the main checkout (no worktree, no replay): **520,963 subagent tokens** (est.
~0.55M) — conformance 288,122 (told not to mutate), tests + mutation 232,841 (27 new mutants + 1 control on scratch
copies, 190 s). Claims re-checked by me before filing. No BLOCKER / MAJOR: no exception scope on any path to
`select_ip_ic` swallows `IPSelectionError` today (the lens traced the 4 call paths and all 15 try/except scopes in
engine code; the replay crashes on an IP error, as intended); the mirror shift, the rule-3 migration, the hovers and
F1's draw gate match the spec.

| ID | Finding | Disposition (LR1 / LR2) |
|---|---|---|
| A01 | REAL survivor, the most serious: the activation sweep's sell branch (`_touches`, anchors by `sd`) — swapped, every sell POI never activates; every activation test is buy-side (the replay would show it: H1 832 is a sell POI) | a sell `_sweep` test |
| R04 / G02 / D01 | survivors: the resolver's graceful anchor scope (`[]`) unpinned; "the locked-fib assert BEFORE the selection" unpinned; `derive_ip_candidates` reading the handed `ip_records` unpinned (output-equal, the D2 contract) | a test each |
| H01 / P01 / M01 / M09 | survivors: asymmetric leg order; the hover TEMPLATE at the 3 POI sites (the tests pinned `customdata`, not the displayed text; self-referential vs the builder) | rendered-hover tests |
| M02 | survivor: no test renders a lens chart WITH a sub POI — `test_m15_sub_poi_hover_site_calls_the_builder` is vacuous (passes with the block skipped) | a rendered lens test replaces it |
| P02 / M03 / P03 / P04 / M04 / M05 | survivors: F1·1 shipped untested; F1's box price levels, its bright / dim rectangle, the M15 fib hover cells 1 / 2 / 10 / 11 | asserts in the rendered lens + H1 tests |
| B29 | the old B mutant is BAD at HEAD (F1 replaced its anchor; F06 / F08 + `test_hover_status_cell` cover it) | re-anchored |
| C1 | §8.5a deviation (1) "no narrow except above it matches, today or later" overclaims "later": `IPSelectionError(RuntimeError)` is caught by any `except RuntimeError` / `Exception`; LANDMINES forbids only `Exception` | docs name `RuntimeError` + a static guard test (LR2) |
| C2 | §8.5 step 3's pin "the resolver on an unstamped `df_probe` (the Phase-2 frame) raises" was built as an MS-level test (`compute_bounded_structure`), undeclared | a `unified_probe` test |
| C3–C9 | stale docs: LANDMINES "Chart gate" line (pre-F1); POI_ZONES_SPEC "M15 fibs are CSV-only" + a false "falls back to `fib_state.cts_idx`" (pre-existing, since 2026-09-29); PART4 ×4 "sub fibs are CSV-only"; FIB_LIFECYCLE_SPEC §9.2's "what still shows"; the `poi_zones` docstring's candle direction (the IP uses the agreed one); §8.5a's "no reader of `versions` left" (2 old review tools `.get` it → empty on new saves) | fixed in place |
| C10 | `export_plotly`'s "[chart][fib] rendered N" counter bumped before the index-membership skip (log line only; equal on the window) | moved after the skip |

**The fix-up (LR1, 2026-10-06c) — as built.** Tests (+9 → 1288): `test_plan_h_landing_b.py` — the sell-side touch
(2 params: inside the zone / above it), the resolver's graceful anchor scope, the candidate rows read from the handed
records, the asymmetric leg order, `_rendered` (a trace's hover as Plotly shows it) asserted at the H1 POI + fib
hovers (incl. F1·1's `%`) and the H1-overlay POI hover, the probe Phase 2 on an unstamped frame (`unified_probe`; the
traceback passes `_run_phase2`), and `test_m15_lens_sub_poi_hover_is_rendered` (a lens chart rendered WITH a sub POI)
in place of the source pin; `test_fib_chart_render.py` — the lens test asserts the fib hover cells 1 / 2 / 10 / 11,
the rendered 61.8 / 80 line, the box's price levels and the one bright rectangle (`_render_m15` takes `poi_zones=`);
`test_institutional_pattern.py` — the locked-fib assert before the selection; new `test_exception_scopes.py` (LR2:
every `RuntimeError` / `Exception` / `BaseException` / bare handler in the engine modules == an allowlist of 8
(module, function) sites — the resolver's 2, the 2 OANDA retries, `run_replay.main`'s export scope, 4 leaf helpers;
none encloses an MS run or the IP selector — + the scan's own self-test). Docs C1 / C3–C9 fixed in place (LANDMINES
×2, POI_ZONES_SPEC ×2, PART4 ×4, FIB_LIFECYCLE_SPEC §9.2, the `poi_zones` docstring, §8.5a ×2); C10 moved.
Mutation (full suite per mutant, `mutant_runner.py`): `mutants_plan_h_review.py` (the lens's 26 + control C00 + my
X01 / X02 — a `RuntimeError` scope at the sub geometry / around the Phase-2 MS run) **28 / 29 KILLED, C00 SURVIVED as
required** (its copy: 1286 passed, 1 skipped); `mutants_plan_h_b.py` **32 / 32** (B29 re-anchored to
`fib_hover_status_text`); `mutants_plan_h_f1.py` **19 / 19**. Prediction for the replay (written before it): 27 / 27
CSVs md5-equal to S_F1; every figure equal (no renderer output changed — C10 is a stdout count, equal on the
window); FETCH GATE PASS; run.log = the 2 `parent_tables` WARNING lines + the 4 `degenerate_parent_cycle` rows, as
before. No save if it holds. **`/compare` vs S_F1 == the prediction:** tests 1288 (OANDA smoke incl.); replay 57.8 s
wall; FETCH GATE PASS; the grep = the 2 WARNING lines + the 4 UNRESOLVED rows + pandas FutureWarnings, `[chart][fib]
rendered 2/3` as before; `cmp_fig_full` 27 / 27 CSVs md5-equal, H1 80 / confluence 274 / counter 135 traces with 0
differing (all attributes), every layout equal; `cmp_hover` 0 changed. Not saved (outputs == S_F1).

### 8.6 Tests

- **New (A), `tests/test_institutional_pattern.py`** — synthetic frames, one rule per test, each written to kill a
  named mutant: the 4 agreed-direction rows; a split candle breaks a run AND is not a candidate (VD drop); a
  volume-only doji IS a candidate; the IC = the last opp candle (another opp candle, or a volume-only opp doji after
  it → not an IC); a true doji after the IC joins the continuation and makes it multi; a true doji INSIDE the opp run
  makes it multi (↑maru · ↓ · ⊖ · ↓IC · ↑big_maru → 3C, never 1B); caps on runs ≥ 4 on both sides with exact
  `pattern_anchor_idx` / `pattern_end_idx` (initial = the most recent `cap`, continuation = the first `cap`); B1a —
  the BOS and CTS anchor candles ARE members (`<` vs `<=`); E1 (an outer-edge doji; the cap-cut case 3464–3466) and E1
  / J1 apply to TRUE dojis only (a volume-only doji at the edge is kept; at the junction it joins its own direction's
  leg); the initial leg needs ≥ 1 sd candle after the slide + trim (1 initial candle + 3 junction dojis at cap 3); a
  continuation of only true dojis fails; the opp run is never cut to the IC; J1 (both placements, the sliding cap, no
  doji into a 3D initial, both rule-A/B branches — below the cap → initial, at the cap → opp — two junction dojis =
  every split); D3c (the cap-1 3D next to the cap-3 stack: 3C wins; 3D when the stack fails; cap 1 ONLY for 3D — a
  2A.ii stays ii; D3c's big reference at the cap-1 `pattern_anchor_idx`); stacking = the same classifier (a stack with
  a maru body-% but a body under the pip floor → pinbar); the big ratio vs the 5 marus before `pattern_anchor_idx`
  with a long maru INSIDE the initial leg (so a reference at the IC or the continuation differs); N2 both ways (a big
  pinbar initial is not big_normal; a big normal continuation is not big_maru); P8a (a 3C whose opp stack is
  pinbar-type passes); the continuation body strictly greater, incl. a float-exact EQUAL body; every rank row 1A–3D
  incl. 3D.ii and one unlisted combination; R1 order over ≥ 3 candidates; TB1 over 3 IC types; TB2a (the user's 4083 /
  4136 example, the 4151 hypothetical, an IC that is the reference's c1 wins, a filled gap skipped as the reference,
  the structural assert on a hand-built instance list) with the tiebreak tuple asserted, not only the pick; the touch
  at both edges (`h == zone_lo`, `l == zone_hi`); `pattern_confirmed_idx` (a cap-hit leg → its last candle; a shorter
  leg → the next candle; an E1-trimmed doji at the cap → the doji; a run cut at the CTS anchor → the next candle).
- **Helpers (A):** `iter_unfilled_imbalances` agrees with `has_unfilled_imbalance` on the existing imbalance
  fixtures; the characterization pins of `find_ic_candidates` (`ic_direction=None`) and `select_ic_variants` (a
  candidate list + a variant dict); `_big_flags` == `classify_big_flags` per candle; `candle_params_for_timeframe`;
  the attrs on the main frame, both lens frames, a sub slice and the probe frame (+ the `KeyError` when missing); the
  qualified copies == today on every row of a fixture; the CSV header on an empty list + the `IP: none` row (≥ 3 rows,
  order pinned); the gate on an active-but-ended record (excluded) and a no-CTS_ESTABLISHED record; the mirror: every
  `_IP_CANDIDATE_IDX_FIELDS` field shifted exactly once PER LENS (a shared object not shifted twice), per-lens
  attribution, the field set pinned from `dataclasses.fields`.
- **B:** the census's list (§8.4) — at least the 10 of 14 `test_poi_activation_moment` tests on
  `_make_second_cts_moment_after_anchor_data` (IC 7 → no pattern) / `_make_multicycle_data` (IC 12 → no pattern) →
  a new fixture with both anchor ≠ moment and a pattern; `test_plan_e_role_pins` (`_poi_history` / `_e4c_poi_history`,
  7 direct `_compute_poi_activation_history(variant_thresholds=)` calls); `test_imbalance_c3_knowability` (`_sweep` ×4,
  the M1 `inners == [1.13]` on a hand-built frame); `test_render_sub_projection` (`> 0` at :515);
  `test_pooled_structure_build._poi_sig` (+ `ip_rank`); `test_zone_proximity`'s fixture meta; the resolver spies in
  `test_e3a_mutation_pins` / `test_ms_cts_update_no_regress`. New: the touch re-check + its reason + the
  `pattern_confirmed_idx` floor; the resolver returns the pick's inner on a fixture where V and the IP DISAGREE (and
  one where the IP picks a non-V IC, its inner asserted); the resolver on an unstamped frame raises (PR1); D2 — `pick
  is not None`, then the drawn pick == the resolver's pick on the final fib, with the REAL collaborator.
- **Mutation tests** (`review_scripts/mutant_runner.py`, full suite): every IP rule branch, and the CALL SITES (the
  pipeline uses the selection; the resolver uses it; the CSV writer keeps the none-rows; the mirror shifts the
  candidates).

### 8.7 Size and cost

- **A:** 1 new module + ~10 files changed + 6 docs (~850 lines, about half tests); one replay; `/compare`; the
  census; `/commit-save`; solo.
- **Between:** three variant replays (~1 min each) + the census run; solo.
- **B:** ~10 files + 8 docs; the re-fixture count from the census (≥ 15 tests); one replay; `/compare`; chart
  review; solo.
- **Reviews:** this plan — done, 879,656 tokens (§8.8); each landing — 2–3 lenses + the mutation run (sizes stated
  then).

### 8.8 Plan review (2026-10-06) — 3 read-only lenses, 879,656 tokens (est. ~0.6M)

Lenses: spec fidelity (222,767), the audit table + code paths (328,693), tests + predictions (328,196); main checkout,
no writes, no replay, no pytest. Claims re-checked by me before filing (marked ✓). Every row is FOLDED into §8.0–8.7
(2026-10-06, after the user's R1–R4 — §6).

| ID | Finding | Disposition |
|---|---|---|
| PR1 | BLOCKER (all 3 lenses) ✓ — `compute_poi_inners_for_cycle` wraps everything in `except Exception: return []` (`poi_zones.py:1168`): under B a missing frame stamp or any IP bug becomes D7's legitimate "no pattern → BOS inner only", invisible to B's `/compare` | B step 3: the `try` keeps only the anchor selection + fib construction; `select_ip_ic` runs outside it; pin: the resolver on an unstamped frame raises |
| PR2 | BLOCKER — B's prediction cannot be exact from a behaviour-neutral resolver shadow + a re-typed trigger recipe: `check_zone_proximity` always includes the BOS inner, reads `poi_active_as_of` PER CANDLE, applies narrow-cycle Rules 2 / 3, scans CTS_CONFIRMED → min(next BOS − 1, reversal − 1); no step computed B's activation histories; after the first moved confirmation a neutral shadow is on the wrong path | §8.4 = variant replays through scratch trees (`review_scripts/replay_variant.py`): V1 the drawn POI only, V2 the MS resolver only, V3 both, each against A's outputs; B's `/compare` == V3 + the format-only deltas; the hand-derived H1 cells (POIs 5 → 3, 710 → 748, 926 → 939) cross-check V3. The resolver shadow stays for per-call logs (PR12, IP time) |
| PR3 | B's output list incomplete: H1 CTS KL zone (1,2) `confirmed_idx` / `confirmation_method`; fib (1,2) `locked_at`; WVMI (1,2) `fp_idx` + both momentums (FP searched to CTS_CONFIRMED + 10); `_final.csv` MS columns from 926; the trigger scan start; the confluence WVMI `triggered_by_*` cells (`_wvmi_trigger_streams_by_lens`); `sub_id` = creation order → a moved record renumbers later subs | all in B's prediction (from V3); M15 diffs keyed on (direction, `starting_idx`) |
| PR4 | The sub mirror of the candidates is unspecified: `LowerTFResult` has no field; a frozen `IPResult` carries no attribution; `_shift_meta_indices` skips `np.int64` (`isinstance(v, int)`); `test_emitter_modules_have_no_unlisted_index_keys[poi_zones]` flags any new `*_idx` dict-literal key in `poi_zones.py` and the new module is not scanned; the `"bos_idx"` / `"cts_idx"` string guard | A: `LowerTFResult.ip_candidates`; records with an attribution meta; a named shift list `_IP_CANDIDATE_IDX_FIELDS`; indices cast with `int()`; IP data in dataclass fields; the static scans extended to `institutional_pattern.py`. B: `ip_tied_count` → `NON_INDEX_INT_META_KEYS`; the new keys in `must_see` |
| PR5 | A changes the save format (24 → 27 CSVs) but lists no docs | A's commit: the `/compare` + `/commit-save` skill tables, WORKFLOWS, `review_scripts/README.md`, POI_ZONES_SPEC (the IP-candidates CSV), memory (counts + a save-format boundary) |
| PR6 | §8.2 rows missing: `poi_per_cycle.py:38` (`versions`); `_wvmi_trigger_streams_by_lens`; the direct `_compute_poi_activation_history(variant_thresholds=)` test calls (`test_plan_e_role_pins` ×7, `test_imbalance_c3_knowability`); ARCHITECTURE.md:294 (cond5 = V30); the charts' `_poi_confirmed_idx` comments; `derive_poi_zones` direct test callers pass no selections | rows added; `derive_poi_zones` keeps an internal selection when none is passed |
| PR7 | Tests: "~15 re-fixtured" is unmeasured — 2 fixtures lose their POI under the IP (`_make_second_cts_moment_after_anchor_data` IC 7, `_make_multicycle_data` IC 12 → 10 of 14 `test_poi_activation_moment` tests need a new fixture); X1's "kept with its tests" is false (0 direct tests of `find_ic_candidates` / `select_ic_variants`); several rule tests cannot kill their mutant as worded | a census plugin under A lists every moved pin (re-pin / re-derive / retire) before B; A adds characterization pins of the V path; §8.6 gets the strengthened rule list (caps on runs ≥ 4 with exact anchors; the BOS / CTS candles as members; a long maru INSIDE the initial leg for the big-ratio pin; a float-exact equal body; both touch edges; both rule-A/B branches; split ≠ candidate; E1 / J1 on true dojis only; a doji inside the opp run → 3C; N2 both ways; P8a; 3D.ii; R1 order over ≥ 3; TB1 over 3 types; TB2a c1 + a filled gap skipped; the tiebreak tuple), call-site pins on fixtures where V and IP disagree, D2 with `pick is not None` first, the CSV none-row, the ended-record gate, the mirror per lens |
| PR8 | Runtime unpredicted: one stack classification = 3.1 ms; B runs the IP on every MS refresh (main, every sub geometry, every Phase-2 iteration); the candidates in attrs are deep-copied by pandas `__finalize__` (1.2 ms / 200 records) | one `classify_candles` call per `select_ip_ic` (every candidate's stacks batched); the record type `__deepcopy__` → self (GOTCHAS, `common/types.py:147`); a timing prediction from V3 |
| PR9 | No reusable single-candle big-flag function — the ratio, thresholds and the `prior_max <= EPS` guard are inline in `classify_big_flags` | A: `_big_flags(length, prior_max, params)` factored out, byte-identical |
| PR10 | §1 item 2's ONE helper not delivered ✓ — `_is_qualified` = `direction == d and vol_dir in (d, 0)` == `direction == d and agreed == d` | A: both qualified copies call `agreed_direction` (byte-identical; wave / WVMI behaviour unchanged, Q-W) |
| PR11 | A's CSV: the touch is applied before `classify_ip`, so the T1 chart-review cases (2375, 4048) are invisible; it can list fib records `derive_poi_zones` skips (no CTS_ESTABLISHED, `poi_zones.py:496`); the prototype check's counts | the CSV also classifies the §3.1 candidates that miss the touch (`touches_zone`, reason `no_touch`, never picked — debug only, D4 holds); `_poi_fib_records` carries the skip; records H1 3 / confluence 20 / counter 7 (the prototype 21 / 8 — sub 3 cyc 3 gated out: active, not locked, ended); the prototype's zone from candle extremes vs `FibState` prices |
| PR12 | The resolver re-picks at every CTS_ESTABLISHED / CTS_UPDATED on the as-of fib, the continuation cut at the as-of CTS — an as-of pick can differ from the final drawn pick; D2 pins only the final fib | stated in §8.1; V3's resolver log counts the calls where they differ |
| PR13 | TB2a: a merged sd gap could straddle the oldest tied IC (the prototype's `max([])` would raise) | ✓ structurally impossible: a c2's candle direction is sd (`imbalance.py:60`), a merged run is contiguous, an IC's candle direction is −sd or 0 → `assert start_idx > oldest_tied_ic_idx` + a hand-built-instance test; no decision |
| PR14 | NITs | `ip_leg_cap_by_tf` via `field(default_factory=…)`; "the cap-`cap` initial"; the last-opp scan bounded at `cts_anchor_idx`; the rank table one data constant; ONE zone-bounds helper for the selection and the re-check (inclusive edges, float); TB2's `evaluated_at=None` stated (cond3's); `candle_params_for_timeframe` keeps `.get(tf, 0.0)` (byte-identical) — only the cap table raises; no re-stamp at orchestrator :928 / :934 (the attrs survive — measured) → a test pins it; the CSV's anchors named like POI meta (`bos_anchor_idx` / `cts_anchor_idx`); params `ic_direction=`, `bos_anchor_idx=` / `cts_anchor_idx=`; `cmp_save.py --names-from <the Week 8 save>`; `cmp_fig_full` exit 1 expected for exactly the 3 NEW lines; no new WARNING lines in run.log; the IP never writes `inst.meta` |
| R1 | ✓ D1b puts an ANCHOR in `first_active`: a continuation that ends below the cap is final only when the next candle closes (conf sub 2 cyc 0: IC 2383, continuation 2384–2385, 2386 down → known at 2386; est 2368 → `first_active` 2385, one candle early). Also sub 3 cyc 0 both lenses (2705 → 2706), conf sub 4 cyc 0 (3390 → 3391); H1 never binds | user: the moment — `pattern_confirmed_idx` (§6 R1) |
| R2 | Names: `ip_leg_sizes` ("size" = a price distance here), `opp_leg_start_idx` (`start_idx` / `end_idx` = lifecycle), `ip_tb_reference_imbalance_idx` (which candle?), `ip_tb_runner_up_idx` | user: renamed — `ip_leg_candle_counts`, `opp_leg_first_idx`, `ip_tb_imbalance_first_c2_idx`, `ip_tb_runner_up_ic_idx` (§6 R2) |
| R3 | B's cond5 reason `ic_outside_fib_zone` (D1a) sits beside cond1's `ic_out_of_fib_bounds` (`poi_zones.py:1012`) | user: `ic_not_touching_fib_zone` (§6 R3) |
| R4 | B's `/compare` against the Week 8 save cannot see B's changes to the IP CSVs (they do not exist there) | user: A ends with `/commit-save` (S_A, local); B's `/compare` against S_A (§6 R4) |

### 8.9 The chart-debugging session — log (2026-10-06 →)

The session the user planned at B's chart review (§8.5a): questions about B's changes, chart aids that make the
differences reviewable (each its own commit, the 27 CSVs byte-identical), then rule experiments one cause at a time —
every output compared against S_B (`20261006_151106_e54f509`) and S_A (`20261006_120513_e1c597c`). The user's
decisions: §6 (the debugging-session block); measurements here.

**F1 — the M15 fib layer + the fib hover (chart aid; F1a–F1f).**
- Found at the cold start: the M15 lens charts drew NO fib (CHARTING_SPEC §8 "H1 chart only — M15 fibs are
  CSV-only"); §8.5b point 1 corrected.
- Census (B's `*_fib_lifecycle.csv`; every record `active` True): confluence 21 records — the old H1 gate (locked, or
  active with the cycle not ended) would draw 20 (not sub 3 cyc 3: 3304 → 3611, ended by `parent_end`, never locked),
  5 of them COLLAPSED: sub 0 c0 (454 → 784), sub 1 c0 (1797 → 1816), sub 1 c1 (1837 → 1898), sub 3 c0 (2639 → 2736),
  sub 7 c0 (4027 → 4086); counter 8 → 7, collapsed sub 3 c0 / sub 7 c0. `IP: none` (all `no_pattern`): confluence
  sub 0 c0 / c1 / c2, sub 1 c0, sub 1 c2, sub 6 c0; counter none. H1: 3 records; sid 1 cyc 1 (689 → 761) collapsed
  (= the degenerate parent cycle (1,1); (1,0) has no fib record). The decided gate (F1b) draws confluence 16, counter
  6, H1 2.
- Built: `_zone_render.fib_record_drawn` (the gate) / `fib_is_bright` / `ip_rows_by_fib_record` (the IP outcome per
  record; sub-keyed on a lens) / `fib_hover_status_text` (the status cell); `export_plotly._add_fib_box` (the one
  drawer); the H1 call site switched; the M15 per-sub block (one legend group `m15_fib`; `fib.lines` default on).
  Docs: CHARTING_SPEC §8 + the M15 sections, FIB_LIFECYCLE_SPEC §9.1 (refined) / §9.2, the M15 module docstring.
- Prediction == the `/compare` vs S_B: 27 / 27 CSVs md5-equal; H1 traces 81 → 80, shapes 235 → 230 (sid 1 cyc 1's
  5 shapes + hover), the two kept fib hovers differ in `customdata` only — `ended (locked)<br>life=[652, 902]
  reversal<br>IP 3A.iii · IC 607` and `active (locked)<br>life=[902, open]<br>IP 3C.ii · IC 832` —, the rest of the
  figure equal; confluence traces / shapes 258 / 178 → 274 / 258, counter 129 / 90 → 135 / 120, each lens == S_B
  exactly once the `m15_fib` group is removed (`plan_h_inputs/fib_aid_check.py <save>`, keyed — `cmp_fig_full` is
  positional), one legend entry
  "M15 fib" per lens. Replay 45.7 s wall. The confluence hovers (status · life · IP; counter = the same cells for
  sub 3 c1–c3, sub 5 c0 `3C.iv · IC 3654`, sub 7 c1–c2):

  | Sub · cyc | BOS → CTS | Status · life | IP |
  |---|---|---|---|
  | 0 · 1 | 454 → 1020 | ended (locked) · [1020, 1169] new_cycle | none (no_pattern) |
  | 0 · 2 | 454 → 1245 | ended (locked) · [1169, 1288] new_cycle | none (no_pattern) |
  | 0 · 3 | 454 → 1721 | ended (locked) · [1288, 1791] new_cycle | 3C.iv · IC 687 |
  | 0 · 4 | 454 → 1794 | ended (locked) · [1791, 1940] reversal | 3C.iv · IC 687 |
  | 1 · 2 | 1934 → 1945 | ended (locked) · [1940, 2231] new_cycle | none (no_pattern) |
  | 1 · 3 | 2078 → 2231 | ended (locked) · [2231, 2343] new_cycle | 1B.iii · IC 2106 |
  | 1 · 4 | 2270 → 2365 | ended (locked) · [2343, 2470] reversal | 2A.iv · IC 2320 |
  | 2 · 0 | 2365 → 2557 | ended (locked) · [2470, 2600] new_cycle | 3A.iv · IC 2383 |
  | 2 · 1 | 2365 → 2609 | ended (locked) · [2600, 2829] reversal | 2A.ii · IC 2442 |
  | 3 · 1 | 2758 → 2842 | ended (locked) · [2829, 2967] new_cycle | 2B.iii · IC 2798 |
  | 3 · 2 | 2758 → 3047 | ended (locked) · [2967, 3589] next_cycle | 1A.iv · IC 2837 |
  | 3 · 3 | 3304 → 3611 | ended · [3589, 3611] parent_end | not evaluated (no POI record) |
  | 4 · 0 | 3304 → 3621 | ended (locked) · [3621, 3819] same_dir_replacement | 1A.iii · IC 3388 |
  | 6 · 0 | 3760 → 4000 | ended (locked) · [3819, 4200] reversal | none (no_pattern) |
  | 7 · 1 | 4027 → 4157 | ended (locked) · [4159, 4177] new_cycle | 2A.iv · IC 4051 |
  | 7 · 2 | 4027 → 4222 | active · [4177, open] (the one bright box) | 1A.iv · IC 4083 |

- Tests: `tests/test_fib_chart_render.py` (21) — the gate (7 cases + the collapsed key), bright, the IP lookup (pick /
  none / the sub key / the duplicate and the unattributed row fail loudly), the status cell (5), the H1 chart (3
  records drawn; a collapsed (0,1) → its 5 shapes + hover gone), a RENDERED M15 lens chart (2 subs: collapsed hidden,
  ended never-locked drawn as "not evaluated", a dead version vanishes, the same anchors on 2 subs keyed apart, one
  legend entry, 15 shapes in the group, only the live version bright — the first test that renders a lens chart from
  sub data) and the toggle off. Suite 1279, all pass. Mutants `plan_h_inputs/mutants_plan_h_f1.py` (19: the gate's
  four terms, bright, the lookup, the status cell, the drawer, both call sites): **19 / 19 KILLED** (5 min).
- **Open — for the rule experiments (not changed):** sub 3 cyc 3 (CTS_ESTABLISHED 3589 → ended 3611 by `parent_end`,
  never locked) was live 22 candles but has NO POI record: the POI record gate (`poi_zones._poi_record_gate`: locked,
  or active with the cycle not ended) skips an ended never-locked record — S_A the same. Whether such a cycle should
  get a POI is a rule question.
- Naming (the parked Naming-Standard audit, register §A): "collapsed" and "degenerate parent cycle" are one test
  under two names (F1c).

**F1·1 — the fib hover's `%%` (chart-aid fix-up, 2026-10-06; the user: "Fix before the save").** Both fib hover
templates printed `61.8%%=` / `80%%=` literally: Plotly's hover template substitutes only `%{…}` and has no `%%`
escape (the bundled plotly.js: `TEMPLATE_STRING_REGEX = /%{([^\s%{}:]*)([:|\|][^}]*)?}/g`). The H1 template had it
since Week 7 (`8c228a4`); F1 copied it to M15. Fixed to `%` (4 labels: `export_plotly.py`, `export_m15_chart.py`).
Prediction == the replay: 27 / 27 CSVs md5-equal to S_B; vs the F1 charts only the fib hover traces' `hovertemplate`
changed — H1 2, confluence 16, counter 6 — the layout (every shape) and every other trace equal. Replay 45.2 s wall.

**R0 — the logged points re-derived on S_S6 (2026-10-09, the resumed session's first step; the kickoff: "measure,
don't copy").** S_B `20261006_151106_e54f509` vs **S_S6** `20261009_020650_5f6bae3`, every fib record keyed (sub_id,
structure_id, cycle_id) — the sub ids are stable (both lenses' `*_subs.csv` byte-equal). PLAN_I §9.9 (measured on its
V3) is CONFIRMED on S_S6, every line. Byte-equal S_B → S_S6: the triggers (both lenses + unresolved), subs, every KL
zone CSV, every WVMI CSV, the H1 imbalance instances — so **§8.5b point 3 (the counter lens, C3) and the H1 trigger
moves 710 → 748 / 926 → 939 stand unchanged**, and H1 `CTS_CONFIRMED` (1,2) stays at 939. The structure-event diffs
are the revamp's own (PLAN_J J1's back-fill event order, the new `cts_frozen` key), none an item-1 point. Every fib
hover gains the limit line (PLAN_I R6-7) and many records a `via` change — format, not a move.

| Point (§8.5b / §8.9) | S_B | S_S6 |
|---|---|---|
| C2 · sub 0 cyc 0 (collapsed) | 454 → 784, IP none | unchanged (hidden) |
| C2 · sub 0 cyc 1 | cross 454 → 1020, IP none (261 · 10 · 18) | single 917 (11-28 11:15) → 1020, life [1020, 1224]: **still none** (50 candidates · 2 touching · 7 patterns, none touching): 929 `ic_not_last_opp`, 930 `continuation_not_big_maru` (926–928 · 929–930 · 931–933, continuation normal); best off-zone 996 1B.iii, 963 2B.i, 1012 3A.iii |
| C2 · sub 0 cyc 2 | cross 454 → 1245, IP none | single 1107 (12-02 10:45) → 1245, life [1224, 1721]: **gains a POI** — 1117 1A.iv (12-02 13:15), legs 3-1-3 (1114–1116 \| 1117 \| 1118–1120), active from 1224 (12-03 16:00) |
| sub 0 cyc 3 / 4 | cross from 454: 687 3C.iv (active 1721 / 1793) | single 1310 (12-04 13:30) → 1721, life [1721, 1793]: 1604 1A.i (12-09 15:00), legs 1-1-1, active 1721 · single 1761 (12-11 06:15) → 1794, life [1793, 1940]: 1772 2B.iv (12-11 09:00), legs 2-1-3 (1770–1771 \| 1772 \| 1773–1775), active from 1793, off / on by imbalance fills at 1860 / 1885, 1888 / 1901, 1921 / 1932 |
| C2 · sub 1 cyc 0 (collapsed), sub 1 cyc 2, sub 6 cyc 0 | IP none | unchanged (sub 1 cyc 2: 3 candidates, none touching; sub 6 cyc 0: 109 · 9 · 10, none touching) |
| **C2 NEW · sub 2 cyc 1** | cross 2365 → 2609: 2442 2A.ii, active from 2608 | single 2561 (12-23 14:15) → 2609 (12-24 02:15), life [2608, 2829]: **IP none** (21 · 6 · 2, none touching): 2565 / 2569 / 2571 / 2575 / 2578 `continuation_not_big_maru`, 2574 `ic_not_last_opp`; best off-zone 2595 3A.iii, 2604 3C.iii |
| **C2 NEW · H1 sid 0 cyc 0** (PLAN_I Q6) | no fib record | single 96 (11-20 22:00) → 430 (12-10 20:00), life [115, 652], DRAWN: **IP none** (150 · 3 · 11, none touching): 177 / 180 / 183 all `continuation_not_big_maru`; best off-zone 206 1A.iii, 386 2A.iii, 131 2B.iv — the first H1 fib hover with `IP: none (no_pattern)` (+ `limit: none (start_trading)`); §8.5a deviation (4)'s "every H1 record has a pick" no longer holds |
| H1 sid 1 cyc 0 (PLAN_I Q4) | no fib record | single 689 → 710, collapsed (hidden), IP none (6 · 5 · 0) |
| T1 (a) · sub 3 cyc 2 (both lenses) | cross 2758 → 3047: 2837 1A.iv, active 3047; without the touch 2992 | single 2915 (12-30 06:45) → 3047, life [2992, 3589] `new_cycle` (was `next_cycle`): **2969 3C.iv** (12-30 20:15), legs 2-2-2 (2966–2967 \| 2968–2969 \| 2970–2971), **TB1 over 2988** (2969 normal > 2988 pinbar; 2 tied), active from **3044** (12-31 15:00). The as-of touch (`fib_zone_bounds` / `ic_touches_zone`, BOS 0.58196, IC 0.57926–0.57944) is False at all 14 CTS events 2992–3043 and first True at 3044 (the `double_maru` CTS update to 0.57500: zone [0.57930, 0.58057]) — the touch still delays it, from at earliest 2992 (12-31 02:00, the CTS-established moment) to 3044 |
| T1 (a) · sub 7 cyc 2 | 4083 1A.iv, TB2 over 4136, active 4222 | unchanged |
| T1 (b) · 2375 3C.iii | sub 2 cyc 0 / 1 | cyc 0 only (2371 / 2372 touch, 2373–2375 miss; outranked by 2383 3A.iv); cyc 1's single 2561 → 2609 no longer contains it |
| T1 (b) · 4048 3C.iv | sub 7 cyc 2 | unchanged (4046 / 4047 touch, 4048 misses) |
| H1 (§8.5b point 4) | 607 · 732 · 832 | unchanged (sid 1 cyc 1's fib `end_reason` `new_cycle` → `next_cycle`) |
| HV (point 5) | — | unchanged; NEW hover lines: `TB1 over 2988 (IC candle type)` (sub 3 cyc 2, both lenses); sub 1 cyc 1's `TB2 over 1842` is CSV-only (a collapsed cycle — its POI is hidden), as in S_B |
| Lifecycle-only moves (pick unchanged) | sub 2 cyc 0 [2470, 2600]; sub 3 cyc 1 [2829, 2967] | [2470, 2608]; [2829, 2992] (both lenses) |
| §8.9 open · sub 3 cyc 3 | no POI record | unchanged (3304 → 3611, life [3589, 3611] `parent_end`, never locked) |

Net on the drawn (non-collapsed) records: confluence no-POI 4 → 4 (sub 0 cyc 2 gains, sub 2 cyc 1 loses; sub 0 cyc 1,
sub 1 cyc 2, sub 6 cyc 0 stay), counter 0 → 0, H1 0 → 1 (the new (0,0)). Tools (scratch, not kept): a keyed
per-record diff of the fib / POI / IP CSVs (`is_collapsed_cycle_zone` on the KL CSV meta) + the as-of touch via the
code's helpers.

**RX1 — the IC reference conditions (rule experiment 1, OPENED 2026-10-09; rounds 1–2 ANSWERED — §6 RX1-B, RX1-1 …
RX1-4, RX1-T, RX1-W1 … W3; the rule DECIDED, below; the implementation plan = **`plans/PLAN_K_ic_reference_condition.md`**
(drafted + cold-reviewed 2026-10-09; K-Q1 … K-Q8 answered; landing A BUILT 2026-10-09 — computed + exported, the
pick unchanged; saved S_K_A `20261009_112137_02e9381`; reviewed + one fix-up (byte-identical); the variant replay V
measured; landing B committed `e2bb77a` — APPLIED + the `IP: none` reason `no_ic_candidate`, §6 RX1-F / RX1-N;
RX1 COMPLETE 2026-10-09 — saved S_K_B `20261009_140048_97795bd` — PLAN_K §11)).** The user:
add IC candidate requirements alongside §3.1, BEFORE the IP ("these new requirements may change the candidates that
make it to the IP check"); found for it (asked "2 or 3 … not wired in"): POI_ZONES_SPEC §3.2's price constraint
(`passes_price_constraint`, no caller, no test) and idx constraint (never coded), both written per the retired
Scenarios 1/2/3 (+ the unwired V30/V60/V90, not chosen). The user chose §3.2's two, re-specified for today's fibs —
verbatim: `plan_h_inputs/ic_reference_conditions_spec.md`. In one line: the IC lies entirely on the near side of the
level the cycle broke out of (sd +1: below), and before the anchor of the pattern that broke it; the level = the
previous cycle's confirmed CTS ANCHOR price (never `cts_threshold`) and, after a reversal for cycles 0…M (the cross
window), also the previous structure's most recent BOS anchor price — when both, the one further opposite the
structure; past M the BOS drops out.
- **The user's H1 example, measured (S_S6):** sid 0's BOS 591 = 0.5736 (== the reversal `bos_frozen` here); sid 1's
  CTS_0 = the CTS_CONFIRMED price 0.57902 (anchor 710; established 0.58195 at 703; thresholds 0.57899 → 0.57856 —
  not used), CTS_1 = 0.57400 (761; threshold 0.57376); M = 2; the reversal pattern (REVERSAL_CANDIDATE) anchor 897 ==
  (1,2)'s CTS_ESTABLISHED pattern anchor; the 1–2 cross hypothetical's bound = (1,1)'s establishment pattern 747 (its
  BOS 728 → candidates 728–746).
- **The census** (`plan_h_inputs/icref_census.py <save> [price]`, the reading in its docstring — refs per fib record,
  the further-opposite choice, the bound, then the best surviving touching pattern; TB2 not recomputed): of the 26 distinct POI
  records (32 lens rows — corrected 2026-10-09 by PLAN_K's review, was "27") 1 pick changes — sub 0 cyc 3 (buy, ref CTS_2 0.57774 = 1245's confirmed anchor): 1604 1A.i (high
  0.57880) is cut by the PRICE; the next best 1447 / 1660, both 1A.iv maru → TB2 decides. The idx bound changes no pick
  (price-only == both). Candidates cut, e.g. sub 0 cyc 3 209 → 85, sub 0 cyc 1 50 → 14, sub 7 cyc 2 90 → 69. No fib
  whose 61.8–80 zone lies entirely beyond its reference (structurally IC-less) — sub 0 cyc 3's zone straddles it.
- **Found while measuring (the open questions):** (Q1) the bound for the BOS reference — "cycle M's establishment
  pattern" is WRONG for M = 0: sub 2 (buy, 2365 → 2557) was established at 2368 (pattern 2367) far below the level and
  crossed sub 1's frozen threshold 0.57926 by raw CTS updates at 2465 = sub 1's reversal pattern anchor; the 2367
  reading wipes every candidate (2383 lost), the "first CTS event beyond P" reading gives 2465 (2383 kept) and equals
  the reversal pattern on all 4 reversal cases (H1 897, sub 1 1936, sub 2 2465, sub 3 2824). (Q2) structures with a
  cross limit but no reversal: sub 4 (`parent_cts_threshold`, P = the parent's CTS_1 threshold 0.57376; anchor 0.57400
  — first wicked at 3521, closed at 3566; P crossed at 3588 by a RAW update, no pattern; = the first M15 candle of the
  parent's H1 breakout 897) and subs 5 / 6 / 7 (`prev_sub_bos_threshold`; sub 7's level, sub 6's BOS 3760 = 0.57806, is
  broken by its cycle-2 establishment pattern 4178). (Q3) P ≠ the anchor on 5 of the 8 limited structures (sub 1
  0.57877 / 0.57882, sub 2 0.57926 / 0.57884, sub 3 0.58156 / 0.58198, sub 4 0.57376 / 0.57400, sub 7 0.57827 /
  0.57806): a previous CTS can sit in the gap — inside the cross window yet past the BOS anchor — sub 3 cyc 1's CTS_0
  0.5818 (2736) already WENT PAST sub 2's BOS anchor 0.58198 (at 2733; not a break — RX1-T), and as written the higher
  (0.58198) is chosen;
  sub 1 cyc 2 the same (0.57877 vs 0.57882); picks unchanged either way. (Q4) cycle 0 with no reference at all (H1
  sid 0, sub 0 — no P): unconstrained or no POI (both have no POI today). (Q5) strict edges (high < ref; ic < anchor —
  the old helper is strict), the IC candle only (legs unchanged), both consumers (drawn + MS's read, D2).
- **The cross-limit breakout watch (RX1-2b), prototyped** (scratch, the M15 frame rebuilt from the S_S6 lens chart):
  the reversal watch's own detector, `BreakoutPatterns.detect_best_for_anchor(A, sd, threshold)`, at every candle A
  whose CLOSE is past the threshold in the structure's direction (the reversal watch opens on such a close-break); the
  first A whose pattern confirms within 5 candles = the break, anchor A. Sub 7 (buy): threshold P 0.57827 → closes
  past from 4183, the patterns at 4183–4187 fail, **4188** `one_maru_opposite` (apply 4189); threshold = the anchor
  0.57806 → **4178** `one_maru_continuous` (apply 4179 — = its cycle-2 establishment). Sub 4 (sell): P 0.57376 →
  **3588** (apply 3589); the anchor 0.57400 → **3587** (apply 3588; 3566 / 3584 close past without a pattern). Subs 5
  / 6: no close past P → no break → the bound passes (RX1-2a). No pick moves under either threshold (sub 7 keeps 4083,
  sub 4 keeps 3388). Round 2 decided W1 = P, W2 = a close past, W3 = fixed, re-armed (sub 7 → 4188; the
  move-to-the-wick variant → 4195, one watch only → none).
- **The rule as DECIDED (2026-10-09)** — ONE bundled condition per §3.1 candidate, applied before the touch / IP, in
  BOTH consumers (the drawn POI and MS's in-flight read, D2), on the fib record (BOS anchor in cycle `a`, target cycle
  `T`, direction `sd`):
  1. **References.** (i) The previous cycle's CTS — when `a ≥ 1`: cycle `a−1`'s CTS_CONFIRMED price (the confirmed
     anchor; never a `cts_threshold`). (ii) The previous structure's BOS or its equivalent — when the structure has a
     cross limit `P` AND `T ≤ M` (or `M` None): a reversal start → the reversing structure's most recent BOS anchor
     price (its BOS_CONFIRMED price); a non-reversal start → the anchor behind `P` (RX1-2: the parent's CTS_{c−1}
     anchor for `parent_cts_threshold`, the previous sub's BOS anchor for `prev_sub_bos_threshold`, …); no `P` → none.
  2. **Choice.** Both → the one further opposite the structure (sd +1 the lower, sd −1 the higher; RX1-3); one → it;
     none → the condition PASSES (RX1-B / RX1-4).
  3. **Price side.** The IC candle lies entirely on the near side: sd +1 `high < ref`, sd −1 `low > ref` (strict).
  4. **Before-the-break side.** The IC idx < the anchor of the pattern that BROKE the chosen reference (RX1-T): (i) →
     cycle `a`'s CTS_ESTABLISHED `pattern_anchor_idx`; (ii) reversal start → the realised reversal pattern's anchor
     (RX1-1); (ii) non-reversal start → the cross-limit breakout watch's break (RX1-2b, W1–W3: a candle closing past
     `P` in direction sd opens a watch; `detect_best_for_anchor(A, sd, P)` confirming within 5 candles = the break,
     anchor A; `P` fixed, every close past re-arms; from the structure start; MS's breakout / reversal pattern set). No
     break reached (yet, or ever) → this side passes (RX1-2a); MS's read sees a break only once its pattern confirmed.
  5. **Unchanged:** the cross window `M` (price going past `P`), the legs, the touch, the IP, TB1 / TB2.
  Assumed defaults (stated to the user 2026-10-09, not objected): strict edges; the IC candle only; the watch a pure
  detector over the structure's frame (no MS state, no structure event) evaluated as of each reader's moment.
  **The anchor behind `P`, per source** (RX1-2 covers every non-reversal start; proposed 2026-10-09, not objected):
  `reversal_bos_frozen` → the reversing structure's most recent BOS_CONFIRMED price; `parent_cts_threshold` → the
  parent cycle c−1's CTS_CONFIRMED price; `prev_sub_bos_threshold` → x−1's latest BOS_CONFIRMED price;
  `prev_sub_cts_threshold` → x−1's confirmed cycle's CTS_CONFIRMED price; `prev_sub_latest_cts` /
  `prev_sub_starting_price` → `P` itself (already an anchor); `none` / `prev_sub_no_cts` → no structure reference.
  The last three of the non-reversal sources do not occur on the reference window (unit tests only); main never has
  a non-reversal start with a `P`, so the cross-limit breakout watch runs on subs only today.
  **Proposed names** (2026-10-09, for the plan): fib record meta `ic_ref_price`, `ic_ref_source` ∈ {`prev_cycle_cts`,
  `prev_structure_bos`, `cross_limit_anchor`} (absent = no reference), `ic_ref_break_pattern_anchor_idx` (the bound),
  `ic_ref_break_confirmed_idx`; candidate reasons `ic_past_reference` (price side) / `ic_not_before_break`; the
  mechanism "cross-limit breakout watch". Housekeeping: POI_ZONES_SPEC §3.2's tables → a history appendix;
  `passes_price_constraint` becomes the price side; `find_ic_candidates`' unread `scenario_context` retired.
  FYI items raised: structurally IC-less fibs (a zone entirely past its reference — 0 on the window; a short
  BOS-to-bound window — sub 1 cyc 2: 1934–1935 only, already no POI) → no POI (D7); the census sees final picks only —
  MS's as-of reads can move where the final pick holds → a variant replay before the switch (as §8.4).

**The IP tiebreaker update (2026-10-09, after RX1; decisions §6 TBU-1 … TBU-5; the spec verbatim
`plan_h_inputs/ip_tiebreaker_update_spec.md`).** Measured on `2d199fd` (= S_K_B's code) by a read-only in-process
pipeline run that wraps `select_ip_ic` — every tie BOTH consumers meet (`plan_h_inputs/tb_imb_measure.py`; the candle
dump `tb_imb_dump.py`; lens indices = sub-local + the sub's slice offset: sub 0 +404, sub 1 +1747, sub 7 +3977):
- **H1: no ties.** M15: three tied records (sub 7's on both lenses); MS's in-flight reads meet the same ties (sub 1
  cyc 1 also on three earlier CTS anchors, local 147 / 148 / 149, the same group), none differs from the drawn read.
- **Confluence sub 0 cyc 3 (a buy; BOS 1310, CTS anchor 1721) — the one record that moves, and the user's
  IC2 / IC3 case.** Tied 1A.iv, both marus: 1447 (h 0.57768) — its gap candle 1448 (the IC is the gap's c1; gap
  0.57768–0.57784) FILLED at 1480 — and 1660 (h 0.57738) — its gap candle 1662 (c1 1661; gap 0.57731–0.57749)
  unfilled through the CTS anchor. Today's last tiebreaker picks 1447 via the far gap 1658 (0.57696–0.57720,
  unfilled). The new tiebreaker: 1447 fails, 1660 alone passes → **1660** (with TBU-2's gap-candle reading; with "the
  whole gap done by IC + 2" 1660 would fail too and nothing would move). After the CTS anchor: 1662 first fills at
  1940 (2025-12-15 03:00) — the first candle back into 1660's range — and 1658 at 1946.
- **Unchanged (no IC passes the new tiebreaker):** confluence sub 1 cyc 1 (a sell): 1842 maru (no sd gap within 3),
  1871 normal (gap 1873 filled at 1877), 1875 maru (gap 1877 filled at 1880) → the type tiebreaker cuts 1871 → the
  last picks 1875, as today; sub 7 cyc 2 (both lenses): 4083 (no sd gap within 3), 4136 (gap 4138 filled at 4145)
  → 4083, as today.
- **Watch at the landing:** MS's sd-proximity read for sub 0 cyc 3 moves with the pick (1660's zone sits ~3 pips under
  1447's) — compare MS's POI reads before touching MS if a structure event moves.
- **COMMITTED `09f056a` 2026-10-09 (one landing, after the chart review):** `poi_zones.immediate_unfilled_imbalance_
  first_c2_idx` (the helper — `iter_unfilled_imbalances` over IC + 1 … min(IC + `POIConfig.ip_immediate_imbalance_
  candles` = 2, the CTS anchor), condition 3's test; it now carries the no-straddle assert the last tiebreaker had) on
  every classified candidate (`IPCandidate` / `IPCandidateRecord.immediate_imbalance_first_c2_idx`, the CSV column
  after `ic_ref_reason`, the mirror's `_IP_CANDIDATE_IDX_FIELDS`); `select_ip_ic`'s chain + the `TIEBREAK_*` names;
  the hover `Tiebreak over <IC>: …`. Tests 1536 → 1537 (10 new pins, 10 rewritten to their purpose under the new
  order — e.g. the candle-type test now gives every IC an immediate gap). **`/compare` vs S_K_B (reuse of the
  44.3 s replay, FETCH GATE PASS) == the prediction** — `plan_h_inputs/cmp_tbu.py` RESULT OK: 22 / 27 CSVs
  md5-equal (every structure-events / KL / fib / WVMI / trigger CSV — MS's read moved nothing); confluence sub 0
  cyc 3 1447 → 1660 (1A.iv, legs 2-1-3 = 1658–1659 | 1660 | 1661–1663, over 1447, its gap 1662, zone 0.57710–0.57738;
  the lifecycle unchanged: active from 1721, ended 1793 `next_cycle`); the relabels on 1875 (never activated — not
  drawn) and 4083 (both lenses); the new column on 120 candidate rows. Charts: counts equal (81 / 236, 135 / 120,
  274 / 264); H1 figure-identical; confluence: the one POI's trace + rect; hovers: that POI, its fib's IP line, 4083.
  Mutation (`plan_h_inputs/mutants_tbu.py`, full suite each): 24 / 24 KILLED — the window bounds, the CTS clamp, the
  fill horizon, the sd filter (also killed by an MS fixture), the assert, the helper's call site, every chain branch /
  runner-up / count / label, the CSV column, the mirror shift, the hover.
- **Chart review (the user, 2026-10-09):** asked why H1 sid 1 cyc 2 did not move from 832 to 860 → no tie there (832 the
  only pattern; 860 `continuation_not_big_maru`, ratio 0.63 vs 0.65) → "Keep it as built" (§6 TBU-6). Also asked
  where the fib hovers sit (an invisible line at the 61.8–80 midpoint, one point per candle from the BOS to the CTS
  anchor — e.g. sid 1 cyc 2 at 0.57816 over 80 candles; the default "closest" hover mode then masks candle hovers in
  that band) → no change now, noted for the chart revamp (register §G).
- **Saved S_TBU `20261009_171255_5b5c307`** (source `5b5c307` = `09f056a` + the close-out docs; save commit `0047646`, trunk `2abc143`; reuse of the 44.3 s replay, FETCH GATE PASS) = `LATEST_week9-entries-decisions` — the IC identification revamp's `/compare` baseline (a save-format
  boundary: the `/compare` skill).

**PAUSED 2026-10-06 (§6 F2a–F2e).** The debugging stops here for the cross-cycle fib revamp (register §D.1 item 2,
a new session). It resumes after the revamp with every point above and in §8.5b as logged, each rule experiment
compared against S_B and S_A (named explicitly). B's landing review (+ F1) opens the revamp session.
Saved (F2d): `20261006_224610_42c137c` = **S_F1** — source `42c137c` (F1 `73e82ed` + F1·1 `811e742` + these docs),
save commit `455241d`, trunk `05b3c1d`; reuse mode of the 45.2 s F1·1 replay, FETCH GATE PASS; its 27 CSVs ==
S_B (only the charts differ: the M15 fib layer, the fib hovers). = `LATEST_week9-entries-decisions`, the revamp's
`/compare` default.
