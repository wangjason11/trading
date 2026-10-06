# Plan H — One POI per cycle: the institutional-pattern IC (Week 9 Part 1 item 1)

**Status: SPEC DECIDED (2026-10-05, after 6 rounds); the implementation plan (§8) REVIEWED and folded (2026-10-06, §8.8 + the
user's R1–R4) — two landings (L1); landing A BUILT 2026-10-06 (§8.3a), its `/compare` == the prediction — paused
before its `/commit-save`.** Living document: every question, answer and decision
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
10. **Tiebreakers among patterns of the same rank + shape (TB, round 4, 2026-10-02 — supersedes the round-3 list and
    R4 / R6):** TB1 IC type maru > normal > pinbar (the IC candle itself; only separates 3C / 3D) → TB2 the most recent
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

## 8. Implementation plan (2026-10-05; REVIEWED + folded 2026-10-06 — §8.8; nothing built)

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
- **Next:** `/commit-save` (S_A, R4) → §8.4.

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
