# Plan I — The cross-cycle fib revamp: one logic for main and sub (Week 9 Part 1 item 2)

**Status: SPEC AGREED — rounds 1–5 ANSWERED (2026-10-07; §6, last R5-1…R5-3): no open question. Unique subs numbered
at their lifecycle START (R4-2); `cross_fib_limit_source` kept; the read rules R4-5 (inclusive) / R4-6 (an ended x−1)
accepted, an ended x−1 yielding its FROZEN thresholds (R5-3); R4-7 withdrawn — the user's ordering argument holds in the
code (R5-1). Session 1 CLOSED 2026-10-07 (docs-only commit). Session 2 (2026-10-07): the implementation plan DRAFTED and
COLD-REVIEWED — §9 (audit, design, stages S1–S5 incl. S3b (R6-1b / R7-1) + S6 after the build (R7-2), predictions by variant
replays, test census, §9.8 the review); round 6
ANSWERED (§6 R6-1 … R6-7) + round 7 (R7-1 … R7-4); the plan APPROVED by the user 2026-10-07 (R7-4); session 3 = the
build: S1 + S2 LANDED (§9.10).** Three sessions (PLAN_H §6 LR3): the user's rules → this spec;
then the implementation plan from a fresh code audit + a cold review; then the build). Living document: every
question, answer and decision is recorded here as it happens (§5 open questions, §6 decisions); measurements in §3.
Register line: `IDEA_PARKING_LOT.md` §D.1 item 2. No engine code changed in sessions 1–2 (session 2's measurement
tools live in `plan_i_inputs/`).

**Input (the user's spec):** `plan_i_inputs/cross_cycle_fib_spec.md` — the user's text VERBATIM (the rule + the
answers to the 8 inputs), canonical, amended in place only as decisions land (each amendment cites its decision ID).
§1 below is the precise engineering reading.

**Baseline:** S_F1 `artifacts/commits/week9-entries-decisions/20261006_224610_42c137c` (= `LATEST_<branch>`).
**Measurement scripts (read-only):** `plan_i_inputs/ref_census.py <save>` (the reference-price census, §3.1);
`plan_i_inputs/prev_sub_census.py <save>` (round 2: the previous-sub rule, the new-cycle threshold check, §3.3).

**Background (superseded as the revamp lands):** `engine_v2/zones/CROSS_CYCLE_FIB_SPEC.md` — §4 the main-only cross
ceiling `M` (built 2026-06-17, §11b Part A, dormant on the window), §3.1 the fill-as-of split, §5 Scenario 1 / revert,
§13 the open items; `FIB_LIFECYCLE_SPEC.md` §6 / §7 (the pre-established cross, Option A's early end); memory
`project_cross_cycle_fib_mode.md`, `project_cross_cycle_fib_unification.md`, `project_sub_single_cycle_fib_pois.md`.

---

## 1. The engineering reading (round 1 — corrected as answers land)

1. **One rule for main and sub — the cross LIMIT.** A structure (an H1 main sid, or a unique M15 sub) has at most one
   reference price `P` (proposed name `cross_fib_limit_price`, §4). A cross fib may TARGET cycle `T` iff no cycle `k < T`
   has a CTS anchor beyond `P` — "beyond" = strictly past it in the structure's direction (sd = +1: price > `P`;
   sd = −1: price < `P`; Q2). The first cycle whose CTS anchor is beyond `P` (`M`, proposed `cross_fib_limit_cycle`) is the
   LAST cycle a cross may target; every cycle after it is single. No `P` → no cross at all (every cycle single).
   - The user's examples: CTS_0 beyond `P` → `M = 0` → all singles; CTS_0 short and CTS_1 beyond → `M = 1` → crosses
     may target cycle 1 (0–1), cycle 2+ single; CTS_0, CTS_1 short and CTS_2 beyond → `M = 2` → 0–1, 0–2 / 1–2, 3+ single.
   - **A cycle's CTS anchor** (the user: "the confirmed CTS anchor and also the most recent CTS_updated"): its
     CONFIRMED anchor (`CTS_CONFIRMED.meta["cts_anchor_idx"]` + `ev.price`); a cycle never confirmed → its LATEST
     anchor (the last `CTS_UPDATED`, else the `CTS_ESTABLISHED` anchor). Measured (§3.2): on the window every cycle with
     a successor was confirmed before the successor was established (22 / 22 pairs), and the only never-confirmed
     cycles are the LAST cycles of conf sub 3 (cyc 3, `parent_end`) and sub 7 (cyc 2, the data edge) — so the fallback
     never decides a cross here; it completes the definition.
   - **Within the limit nothing else changes** (the user): the dead-cycle walk (contiguous live prior cycles → the
     earliest start `earliest_x`), the own-imbalance test, the version transitions (extend / shrink → v+1 / revive in
     place / `cross_failed` → single), the activation conditions and the lifecycle. `M` is a ceiling, never a producer
     (unchanged from CROSS_CYCLE_FIB_SPEC §4.3: a target ≤ `M` still crosses only if the walk finds a live prior cycle).
   - This IS the main-only ceiling of CROSS_CYCLE_FIB_SPEC §4 (`_cross_allowed_for_target`), generalised: (a) to subs
     (today unbounded); (b) with the reference chosen by the STARTING TRIGGER (below; main's today = the previous
     structure's last BOS KL-zone max-expanded outer, `_get_prev_bos_outer`); (c) strict "beyond" (today inclusive
     `>=` / `<=`, Q2); (d) the anchor fallback above (today an unconfirmed cycle is skipped).
2. **The reference `P` by the structure's starting trigger** (main: the trigger that started the sid; sub: the FIRST
   trigger in real time mapped to the unique sub = `started_by`, since the pool builds the whole unique sub from it):

   | Started by | `P` (the user's words → the reading) |
   |---|---|
   | main: the first MS run ("start trading", sid 0) | none → no cross fibs |
   | main: reversal | "the most recent bos_threshold that a valid pattern broke out of that triggered the reversal" → the APPLIED reversal's `bos_frozen` (`REVERSAL_CANDIDATE.meta["bos_frozen"]` of the reversing sid — in the code read off the same reversal's
`STATE_CHANGED(to=reversal).meta["bos_frozen"]`, equal by the watch, asserted [session 2, §9.3 D2; the assert compares the OPEN WATCH's frozen barrier — S2's landing review: a later anchor's reversal winner can apply with no candidate at the realised apply]; the MS barrier the reversal pattern broke — an earlier failed close-break `rv_anchor_failed` moved the threshold, not this) |
   | sub: reversal | the same, from the REVERSING sub's applied `REVERSAL_CANDIDATE` |
   | sub: first_confluence on parent cycle 0 | none → no cross fibs |
   | sub: first_confluence on parent cycle c ≥ 1 | "the most recent CTS_threshold that a valid pattern broke from to result in CTS established" → the PARENT's cycle c−1 `cts_threshold` as of parent (S, c)'s `CTS_ESTABLISHED` — c−1's confirmed CTS price, moved by its `CTS_THRESHOLD_UPDATED` range syncs [Q3]. No `cts_frozen` VARIABLE exists, but the process mirrors the reversal [R3-1, §3.3]: the new cycle's breakout must CLOSE beyond the threshold, the threshold is FROZEN while the detected pattern is pending (anchor → apply) and a failed break moves it (`range_sync`) — so this is the frozen threshold the breakout broke (H1 (1,2): 0.57376, anchor 897 → apply 902) |
   | sub: first_counter / subsequent_confluence / subsequent_counter | [R2-1, supersedes Q1] at the unique sub's `start_idx` (the moment), the PREVIOUS unique sub "x−1" (= `sub_id` − 1 — the user's (i); unique subs are numbered at their lifecycle START [R3-2, accepted R4-2], so x−1 has always started by x's start): **same direction** as x → x−1's `cts_threshold` of its most recent CONFIRMED CTS (that cycle's confirmed CTS price, moved by its `CTS_THRESHOLD_UPDATED` range syncs; a cycle only established / updated is never used); **opposite** → x−1's current `bos_threshold` (of its most recent BOS); opposite with no `bos_threshold` yet → x−1's finalized `starting_idx` price (= its BOS_0 price, §3.3) [R2-2]; same direction with no confirmed CTS yet → x−1's LATEST CTS anchor price at x's start (established / updated) [R2-b (b)]; no x−1 → no `P`. x−1 is read INCLUSIVE of x's start candle (its events whose moment ≤ x's `start_idx`) [R4-5]; an x−1 that has already ENDED is read as of its `end_idx` — its geometry runs on to the data edge, and those post-end events never count [R4-6]; it is read by MOMENT, so an x−1 ended by a reversal or a new parent cycle yields its FROZEN thresholds — the applied reversal's `bos_frozen`; a pattern or watch pending at the end → the value it holds [R5-3] |

   `P` is determined ONCE, at the structure's start (main: the reversal's apply; a sub: its `start_idx`), and never
   changes [R2-4].

3. **The user's answers to the 8 inputs (§6 I1–I8):** variants attach to TRIGGERS, not to sid; "Scenario X" names
   retired (I1); the pre-established cross runs on main too, inside the limit (I2); post-reversal fib handling on
   main = the sub's after a reversal-started sub (I5); no single + cross coexistence (I8); round 1 settled the rest
   (Q4 = I3, Q5 = I4, Q8 = I6, Q7 = I7).

4. **The rule as decided after round 1 — one logic for every structure (main sid, unique sub):**
   - **Cycle 0 is always single** [Q4, Q6]: main's start-trading cycle 0 gets one (today "simple flow": none); a
     post-reversal cycle 0 gets one (today only under Scenario 1); a sub's cycle 0 as today — incl. its late
     activation on a `CTS_UPDATED` (FIB_LIFECYCLE_SPEC §6, "activated_on: update"), now on main too [I5].
   - **A cross may target cycle T ≥ 1 iff the structure has a `P` and no cycle k < T has a latest CTS anchor STRICTLY
     beyond `P`** [R0, Q2]; then, exactly as the sub today: the dead-cycle walk with the fill checked to the CURRENT
     candle [Q5], the own-imbalance test, the version transitions, the pre-established phase (target T evaluated during
     cycle T−1's tail on `CTS_THRESHOLD_UPDATED`, with Option A's early end of cycle T−1's fib) [I2], and `cross_failed`
     → a single in the established phase [I5]. Never a single and a cross on one cycle [I8].
   - **No Scenario 1, no revert** [Q4]: main's outer gate and `_cross_cycle_data["cycle0"]` decision path go.
   - **One fib decision for the drawn fib and MS's in-flight POI read** [Q7] — MS needs `P` (main: its own reversal's
     `bos_frozen`; a sub: from the pool, fixed by its first trigger); the in-flight read covers every target ≤ `M`
     (CROSS_CYCLE_FIB_SPEC §11b "Part B", now required).
   - **One meta format** [Q8] — main's cross records carry the sub's keys.
   - **`P` is fixed at the structure's start** [R2-4]; the previous-sub rule reads sub x−1 at x's `start_idx` [R2-1] (inclusive [R4-5]; an ended x−1 as of its `end_idx` — its frozen
     thresholds [R4-6, R5-3]; ids numbered at the start [R3-2]);
     its fallbacks: R2-2 (opposite, no `bos_threshold` yet → x−1's starting price), R2-b (same, no confirmed CTS →
     x−1's latest CTS anchor price). (Round 1's edge — "an opposite sub with no `bos_threshold` → no `P`" — is superseded by R2-2.)
   - **Fib activeness stays clamped by the lifecycle — unchanged** [R2-5] (FIB_LIFECYCLE_SPEC §15.3 / §15.4):
     `start_idx` = max(own first-active, the STRUCTURE lifecycle floor), `end_idx` = the earliest terminal. The one
     documented difference from zones: the clamp is to the STRUCTURE floor, not the cycle's `CTS_ESTABLISHED` moment,
     because a pre-established cross legitimately starts before its own cycle is established (§6) — with I2 that now
     applies on main too.
   - **A sub's cycles established before its `start_idx`** (the probe builds the structure back to `starting_idx`;
     e.g. conf sub 1 cyc 1, established 1894, start 1940) take their fib decision with the same `P`: the unique sub is
     built whole at its start, and the clamp keeps every fib inactive before it. The same holds for an H1 reversal sid
     [session 2, §9.2 F1 / F3]: sid 1's cycles 0 / 1 (established 703 / 748) precede its start 902, and its cross
     pre-established at 741 uses the `P` fixed at 902.

## 2. What changes (round-1 reading, from the census §3 — nothing built; still valid after rounds 2–5: the x−1 rule gives the same `P` on the window, §3.3)

- **Subs:** a cap where there was none. On the window 6 of today's 8 sub cross records become singles (conf sub 0
  cyc 1–4: no `P`; sub 2 cyc 1: `M = 0`; sub 3 cyc 2: `M = 1`); sub 7's two stay (Q1 (a): `P` 0.57827, `M = 2`).
- **Main:** sid 1's `P` = the reversal's `bos_frozen` 0.5736 = today's KL-zone `P_rev` → `M = 2`, as today. The
  changes come from I2 / I5 / Q4–Q6 / Q8: two NEW H1 cycle-0 single fibs — sid 0 cyc 0 (BOS 96 → CTS 430; live: a
  new fib — CORRECTED session 2 (§9.2 F2): no POI on the window (no IP pick), and it can never move (0,0)'s
  confirmation, since cycle 0 never confirms by proximity) and
  sid 1 cyc 0 (689 → 710; collapsed — it ends before the reversal applies at 902 — so hidden); the pre-established
  cross on main (CORRECTED session 2, §9.2 F3: ONE on the window — (1,1)'s cross pre-established at 741 during
  (1,0)'s tail; collapsed); the (1,1) cross's meta (Q8). Measured by variant replays in the
  plan session.
- **The MS in-flight read (Q7)** — the 8 sub cycles where MS's snapshot reads the single fib while the drawn POI
  comes from a cross (D5) are EXACTLY today's 8 sub cross records: 6 align by themselves under the rule.
- **The cascade** (as PLAN_H's): a different fib → a different 61.8–80 zone → a different IP pick (or none) → MS's
  sd-proximity CTS confirmation → the structure → the zone-proximity triggers → the subs. Predicted by variant
  replays in the plan session, one cause per `/compare`.
- **PLAN_H's paused debugging points move:** sub 0's no-POI cycles (§8.5b C2) and sub 3 cyc 2 (the T1 example) are
  among the cycles whose fib changes — re-derive them after the revamp.

## 3. Measured (S_F1 `20261006_224610_42c137c`)

### 3.1 The reference-price census (`plan_i_inputs/ref_census.py`; times UTC, M15 i = H1 i // 4 + 15 min × i % 4)

| Structure | Started by (first trigger) | `P` | CTS anchors (beyond `P`?) | `M` | Today → under the rule |
|---|---|---|---|---|---|
| H1 sid 0 (buy) | start trading | none | — | — | cyc 0 no fib, cyc 1 single → no cross either way |
| H1 sid 1 (sell) | reversal (watch 897, apply 902) | 0.5736 (`bos_frozen`) | c0 710 (12-29 12:00) 0.57902 no · c1 761 (12-31 15:00) 0.57400 no · c2 905 (01-09 15:00) 0.57112 yes | 2 | = today's `M` (`P_rev` 0.5736) |
| conf sub 0 (buy, 454 11-21 15:30) | first_confluence (0,0) at 463 | none | c0 0.57320 · c1 0.57520 · c2 0.57774 · c3 0.58246 · c4 0.58314 | — | c1–c4 cross from c0 → **single ×4** |
| conf sub 1 (sell, 1797) | reversal of sub 0 (apply 1940, 12-15 03:00) | 0.57877 | c0 1816 0.58051 no · c1 1898 (12-12 16:30) 0.57877 **EQUAL** · c2 1945 0.57656 yes | 2 (strict; `>=` → 1) | c1–c4 already `cross_failed` singles → unchanged |
| conf sub 2 (buy, 2365) | reversal of sub 1 (apply 2470, 12-22 15:30) | 0.57926 | c0 2557 (12-23 13:15) 0.58434 yes | 0 | c1 cross from c0 → **single** |
| sub 3 (sell, 2639; both lenses) | reversal of sub 2 (apply 2829, 12-29 09:15) | 0.58156 | c0 2736 0.58180 no · c1 2842 (12-29 12:30) 0.57902 yes | 1 | c2 cross from c1 → **single** (c1, c3 `cross_failed` singles) |
| conf sub 4 (sell, 3304) | first_confluence (1,2) at 3611 | 0.57376 (Q3 reading) | c0 3621 0.57112 yes | 0 | one cycle → unchanged |
| counter sub 5 (buy, 3621) | first_counter (1,2) at 3759 | 0.58105 (Q1 (a) = (b)) | c0 3760 0.57806 no | not reached | one cycle → unchanged |
| conf sub 6 (sell, 3760) | subsequent_confluence (1,2) at 3819 | (a) 0.57112 · (b)/(c) 0.58105 | c0 4000 0.57255 | — / 0 | one cycle → unchanged |
| sub 7 (buy, 4027; both lenses) | subsequent_counter (1,2) at 4159 | (a)/(b) 0.57827 · (c) 0.57112 | c0 4086 0.57674 no · c1 4157 0.57798 no · c2 4222 (01-19 21:30) 0.58034 yes (never confirmed) | 2 / 0 | (a)/(b): c1, c2 crosses unchanged; (c): both → single |

Q1's readings in the table: (a) the most recent OPPOSITE-direction sub started at or before the trigger; (b) the
previous sub on the trigger's lens; (c) the sub this one replaces (`same_dir_replacement`) — each sub's MS
`bos_threshold` as of the trigger candle.

- **The equality on conf sub 1 is structural, not a coincidence:** sub 0's last `bos_threshold` was lowered by a
  probe wick at 1898 (`BOS_THRESHOLD_UPDATED` 0.57877, `probe_no_break` — the wick crossed, the close did not), and
  1898 is sub 1's cycle-1 CTS anchor (sub 1 IS sub 0's reversal structure: its CTS lows are the probes of sub 0's
  barrier). The break came later: the close below at 1936 (watch, `bos_frozen` 0.57877) → applied 1940, inside sub 1's
  cycle 2. Strict "beyond" makes cycle 2 (the break) `M`; inclusive would make the probe cycle `M` (Q2).
- **Timing ≠ price (Q4):** conf sub 2's cycle 0 was ESTABLISHED at 2368, before sub 1's reversal applied at 2470 —
  Scenario 1 (a timing test: CTS_0's moment ≥ the reversal's apply) would say FALSE (no cycle-0 fib; cycle 1 decides
  cross vs single) — yet its CTS anchor 2557 (0.58434) later ran past `P` 0.57926 → `M = 0` under the rule (all single).
- **Main sid 1's `P` equals today's `P_rev`:** the KL BOS zone of sid 0 cyc 1 has outer 0.5736 = its INIT bounds (the
  BOS price; no expansion) = the MS `bos_threshold` = `bos_frozen` at the reversal watch 897. I expect "the last BOS
  zone's max-expanded outer == the reversal's `bos_frozen`" by construction (a BOS zone's INIT outer is the BOS price;
  its expansions are the `BOS_THRESHOLD_UPDATED` moves) — to verify in the plan's code audit; the rule reads
  `bos_frozen` directly either way (one source for main and sub).

- **H1 sid 1's gap liveness, re-measured on today's code** (`has_unfilled_imbalance`, sell, fill 0.70, asked at t):
  cycle 0 [689, 710] unfilled at t = 748, 761, 810, 880, 902 (all True); cycle 1 [728, 761] True at 748 / 761 / 810,
  **False at 880 and 902**. So (Q5) the "current" check passes wherever the snapshot does on the window; and (I2) the
  only `CTS_THRESHOLD_UPDATED` in (1,1)'s tail is at 880 (0.57376, a range sync), where cycle 1 is already dead → the
  walk from target 2 stops at cycle 1 → predicted: no pre-established cross on main on the window (to confirm by the
  plan's variant replay). [CORRECTED session 2, §9.2 F3: target 2 forms none, as predicted — but target 1 does, at
  741 (12-30 19:00), during (1,0)'s tail; this check had looked at target 2 only.]

### 3.2 "A cycle's CTS anchor" in the code today (the user's question)

- **Every CTS event carries the cycle's CTS anchor AS OF that event** (GLOSSARY "Naming Standard": `cts_anchor_idx` =
  a location, never a moment; `ev.price` = that anchor's price; read through `ef.cts_anchor_idx(ev)`):
  `CTS_ESTABLISHED` → the first anchor; a pattern-path `CTS_UPDATED` → the new anchor in `meta["cts_anchor_idx"]`; a
  raw-path `CTS_UPDATED` (`via: replay_raw`) → its `ev.idx` IS the new extreme (no meta key); `CTS_CONFIRMED` → the
  FINAL anchor = the last update's (e.g. H1 sid 1 cyc 1: established 748, raw updates 751–760, pattern update 761
  0.574, confirmed at 810 with anchor 761).
- So "the CTS anchor" is a moving value until the confirmation; the confirmed anchor is the latest one at the
  confirmation. Consumers: the fib's `cts_idx` follows every update until it locks at `CTS_CONFIRMED`
  (FIB_LIFECYCLE_SPEC §2); FibTracker's `_cts_by_cycle` (the dead-cycle walk's windows AND today's ceiling `M`) is
  written ONLY at `CTS_CONFIRMED` — an unconfirmed cycle is skipped by the ceiling (= treated as not beyond).
- Measured: every next cycle is established after its predecessor's `CTS_CONFIRMED` (H1 3 / 3 pairs, confluence 14 /
  14, counter 5 / 5) → at every cross decision the prior cycles' anchors are confirmed (final).

### 3.3 Round 2 (2026-10-07; `plan_i_inputs/prev_sub_census.py <save>`, read-only, on S_F1)

**The previous-sub rule (R2-1) on the window** — the three subs it governs; x−1 = `sub_id` − 1 (= start order here):

| Sub x (dir, started by, `start_idx`) | x−1 (dir, live) | Branch | `P` (source event) | vs round 1 (Q1 (a)) |
|---|---|---|---|---|
| counter sub 5 (buy, first_counter, 3759 01-13 01:45) | conf sub 4 (sell, 3621–3819) | opposite → `bos_threshold` | 0.58105 (`BOS_CONFIRMED` 3306, cyc 0) | same — not reached |
| conf sub 6 (sell, subsequent_confluence, 3819 01-13 16:45) | counter sub 5 (buy, 3759–4159) | opposite → `bos_threshold` | 0.57112 (`BOS_CONFIRMED` 3656, cyc 0) | same — one cycle |
| sub 7 (buy, subsequent_counter, 4159 01-19 05:45) | conf sub 6 (sell, 3819–4200) | opposite → `bos_threshold` | 0.57827 (`BOS_THRESHOLD_UPDATED` 3806, cyc 0) | same — `M = 2`, crosses kept |

(Labels, round 3: "conf" / "counter" in front of a sub = the lens chart(s) it is DRAWN on, never a separate sub — a
unique sub is lens-independent: one sub 5 (its records all come from counter-lens triggers, so only the counter chart
shows it), one sub 6 (confluence only), one sub 7 (both). Sub 4 is not in the table: it is started by a
first_confluence (parent (1,2) at 3611, 01-09 12:45) → Q3, `P` 0.57376 (the frozen threshold below), its cycle 0
anchor 0.57112 is beyond → `M = 0`, one cycle.)

- **Every x−1 on the window is the opposite direction** → R2-1 changes no window output vs round 1; the same-direction
  branch is unexercised. x−1 crosses lenses (sub 6, confluence ← sub 5, counter-only), as Q1 (a) did.
- **Same-direction branch, illustrated** (NOT applied — sub 4 is a first_confluence): sub 4 (sell, start 3621)'s x−1 is
  sub 3 (sell, ended 3611 `parent_end`). At 3611 sub 3's cycle 3 is established (3589, 01-09 07:15, 0.57318) but never
  confirmed → the rule reads cycle 2's threshold 0.57376 (`CTS_THRESHOLD_UPDATED` 3521, 01-08 14:15) — a level sub 3's
  own unconfirmed cycle 3 had already passed (the consequence of "confirmed only").
- **The reversal-started subs agree with the x−1 rule** (3 / 3): the reversing sub IS x−1 (subs 1 ← 0, 2 ← 1, 3 ← 2)
  and x−1's `bos_threshold` at the apply = the applied `bos_frozen` (0.57877, 0.57926, 0.58156) — MS invariant 4: the
  threshold cannot move during a watch. Not guaranteed in general (another sub can start between the reversing sub
  and its successor), so the reversal rule stays as decided.
- **The R2-2 fallback = x−1's BOS_0 price**: on all 8 subs BOS_0's anchor == `starting_idx` (sub 0 454 0.55874; 1 1797
  0.58244; 2 2365 0.5736; 3 2639 0.58478; 4 3304 0.58105; 5 3621 0.57112; 6 3760 0.57806; 7 4027 0.57362) — the value
  `bos_threshold` takes once x−1's cycle 0 is established. Every sub's CTS_0 was established before its own
  `start_idx` (8 / 8) → on the window x−1 always had a `bos_threshold` at x's start (the user: "shouldn't happen").
- **id order vs start order**: equal on the window (start 1020, 1940, 2470, 2829, 3621, 3759, 3819, 4159 for ids
  0–7); they can differ in general (R2-a).
- **The CTS-side freeze — CORRECTED in round 3 (R3-1).** Round 2 said "MS has no frozen CTS barrier" and that the
  detectors use the threshold "only as an alternative validity route" — the second part was WRONG (only the `c1`
  tail checks were read). Correct reading: there is no `cts_frozen` VARIABLE or event, but the PROCESS mirrors the
  reversal watch, as the user expected:
  - **The break is required:** every breakout detector starts with `check_break(c0, break_threshold, direction)` —
    the pattern's FIRST candle must CLOSE beyond the threshold (two of them also need a share of the body beyond,
    `body_check`); `break_threshold` = `breakout_th` = the active range's breakout bound (`_range_breakout_threshold`),
    which `cts_threshold` mirrors (`_sync_thresholds_from_range`). After a confirmation the range stays active until
    the next breakout (its only `_deactivate_range` is `range_breakout`), so every new cycle faces it.
  - **Frozen while the pattern is pending:** a breakout found at anchor `i` applies at its apply candle (≤ `i` + 5);
    the candles in between are back-filled with `freeze_range=True` (`_replay_step_no_patterns`: the range does not
    expand) → the threshold holds its anchor value until the apply. Window, H1 (1,2): the `continuous` pattern's
    first candle 897 (01-09 07:00) closes 0.57334 beyond `cts_threshold` 0.57376; 898–901 print new lows (0.5732,
    0.57283, 0.5728, 0.57276) yet `range_lo` = `cts_threshold` stays 0.57376 to the apply at 902 (01-09 12:00) —
    `CTS_ESTABLISHED` (1,2), anchor 902 0.57228.
  - **A failed break moves it** (the `probe_no_break` / `rv_anchor_failed` analogue): a candle beyond the threshold
    with no valid pattern is absorbed — the range expands, `CTS_THRESHOLD_UPDATED` (`range_sync`). Window, H1 (0,0):
    candle 650 (12-24 00:00) wicks to 0.58448 past 0.58434 but closes 0.58406 inside → threshold 0.58448; 651 (01:00)
    closes 0.58488 beyond it → `one_maru_opposite` anchored 651 → `CTS_ESTABLISHED` (0,1) at 652.
  - **The differences:** the reversal's freeze is an explicit WATCH (state `reversal_watch_active`, barrier
    `reversal_bos_th_frozen`, an expiry, events `REVERSAL_WATCH_START` / `REVERSAL_CANDIDATE` carrying `bos_frozen`);
    the breakout's is implicit in the step (the pattern is evaluated at its anchor with ≤ 5 candles of look-ahead; no
    state, no event records the frozen value). Measured: 17 / 17 new cycles (H1 3, subs 14) have their establishing
    CTS anchor beyond c−1's `cts_threshold` at the moment.
  - **So Q3's `P` = the frozen breakout threshold** (c−1's `cts_threshold` just before MS resets it at the apply;
    sub 4's 0.57376 IS the frozen value of H1 (1,2) above). For the plan session: stamp it on `CTS_ESTABLISHED` meta
    (cycle ≥ 1) under a name mirroring `bos_frozen` — e.g. `cts_frozen` — so the first_confluence `P` is READ, as the
    reversal's is.

### 3.4 Round 3 (2026-10-07): when a unique sub gets its `sub_id` (the user's R2-a question)

Read from `lifecycle_sweep.py` (`_phase0_fire` → `_resolve_and_record`, `_phase1_record_start`, `_phase2_sub_start`)
and the S_F1 triggers CSVs:

- **At the FIRST trigger's fire moment** (phase 0, `trigger_idx`) the sweep runs the probe, builds the geometry and
  calls `SubStructurePool.get_or_create` → the unique sub EXISTS and takes the next `sub_id`, with
  `start_idx = None`. The probe runs OFFLINE: it may look past the fire moment to its finalize.
- **The TriggerRecord is created at the same moment with its `start_idx` already filled** —
  `max(probe_finalize_idx, trigger_idx, parent_floor_idx)`, known in advance from the offline probe — and its
  RECORD_START is queued at that index. It is never NULL.
- **The sub's `start_idx` is set at that later moment** (phase 2, SUB_START).
- Conf sub 0 (the only long gap on the window): the first_confluence on parent (0,0) fires at 463 (11-21 17:45);
  its probe finalizes at 1020 (12-01 13:00, `second_cts_reached` = the sub's cycle-1 CTS established) → from 463 to
  1019 sub 0 exists with `sub_id` 0 and `start_idx` None; its record has `start_idx` 1020; sub 0 starts at 1020. In
  live real time nothing about sub 0 is knowable before 1020 (the probe has not finalized) — the replay only knows it
  early because it resolves the probe offline.
- **Which triggers can lag** (CORRECTED round 4 — round 3 said "only a first_confluence"): `start_idx` = max(finalize,
  trigger, parent floor), so there are two sources. (1) A first_confluence's Phase 2 probe (finalize after the
  trigger); the sibling types are asserted `finalize == trigger_idx` (a cache hit inherits an earlier finalize) and a
  reversal starts at its `R`. (2) The PARENT FLOOR, for ANY trigger type: a trigger on a reversal structure's cycle,
  fired before that structure started, starts at the floor. Window: the 4 unresolved triggers on H1 sid 1's cycles
  0 / 1 (first_confluence 2815 and 2995, first_counter 3283, subsequent_confluence 3487) fired before sid 1's floor
  3611 — dropped only because those cycles end AT 3611 (degenerate); had one stayed live, each would have started at
  3611. Window lags (resolved records): sub 0 463 → 1020 (557 M15 candles); sub 4 3611 → 3621 (10); every other
  record starts at its trigger (sub 2's FC record at 2611 is a cache hit, finalize 2470).
- **So ids follow FIRE order, lifecycles follow START order.** They disagree only if a NEW sub fires AND starts inside
  a lagging sub's fire → start gap (a first_confluence's, or any floor-lagged one — round 4): fire(x−1) < fire(x) ≤ start(x) < start(x−1). Then at x's start the
  literal x−1 has not started (in real time it does not exist yet), and a live engine would have numbered them the
  other way round. Never on the window: no trigger fired inside 463–1020 or 3611–3621; id order == start order (8 / 8). (Round 5: within one parent structure it cannot happen among subs that start;
  the one remaining case crosses a main reversal — below.)
- **R3-2 — ACCEPTED round 4 (R4-2).** As proposed: keep the user's (i) — literal `sub_id` − 1 — and make it true by construction: number each unique
  sub at its START (phase 2), not at its first trigger's fire (or renumber by `(start_idx, creation order)` at the
  sweep's end), with an assert that `sub_id` order == `start_idx` order. Window: no id changes (byte-identical
  expected). For the plan session: a sub whose records are all zero-length never starts — it would get no id (today
  it holds one); `sub_id` is the join key on every sub event / zone / CSV row (identity, PART4 §17), so this is a
  contract-level change — atomic, in one commit.

**Round 4 (2026-10-07; read from the code, measured on the S_F1 subs / triggers / events CSVs):**

- **A first_counter CANNOT start before its first_confluence sibling** (CORRECTED round 5 — round 4 claimed it could;
  the user's argument, verified in the code [R5-1]):
  (1) the first_counter fires at its parent cycle's FIRST sd zone-proximity (`uc1_trigger` ←
  `orchestrator._first_sd_prox_gate`), and the proximity scan STARTS at the cycle's `CTS_CONFIRMED` moment
  (`zone_proximity.py`: `scan_start = event_moment(CTS_CONFIRMED)`); its probe ends at its trigger, so it starts there
  (or at the shared parent floor);
  (2) the first_confluence fires at the cycle's `BOS_CONFIRMED` (== its `CTS_ESTABLISHED` moment, asserted in
  `parent_tables.py`) and its probe's SUPREME bound is the cycle's confirmed CTS ANCHOR, price-mapped into that
  anchor's H1 hour (`first_confluence_trigger.py`; `unified_probe`: every finalize ≤ `probe_end_idx`) — the anchor
  comes at or before its confirmation.
  So FC start ≤ the confirmation hour ≤ the first_counter's start: an sd-proximity confirmation → at most the same
  hour (a tie → the FC fired first → the lower number); a pullback confirmation → earlier. Round 4 missed the probe
  bound (sub 0: bound = H1 (0,0)'s confirmed anchor 430 → ~M15 1720; finalize 1020; no first_counter before the
  confirmation at H1 439). The other paths round 4 listed do not open either: at the FC's fire the previous parent
  cycle's records end on the same candle (cycle end = the next cycle's floor, `parent_tables.py`; sub 2's (0,0) record
  ends 2611 = (0,1)'s FC 2611; sub 3 ends 3611 = sub 4's FC), so no sub is live during the wait and no sub reversal can
  spawn; a later cycle of the same structure starts at or after this cycle's end, and an FC sub that starts at all
  starts before that end. What CAN land in the wait: the parent cycle ending → the FC record is zero-length → the sub
  never starts (R4-4).
- **A numbered sub that never starts is allowed today** (the user: "if it’s the case there are unique subs numbered
  that never starts, then that shouldn’t be happening even now"). The id is taken at fire (step 3, the geometry),
  before step 6 decides whether the record is ever live, and `_after_sweep` (`lifecycle_sweep.py` ~523–530)
  explicitly accepts "no `start_idx`, no live records". A sub's FIRST record is zero-length when (a) its parent cycle
  ends at or before the record's start (an FC whose probe finalizes after the parent's next cycle is established), or
  (b) the sub's own M15 structure reverses at or before it. E.g. sub 0: had H1 (0,1) been established before 1020 (it
  came at 2611), sub 0 would hold id 0 and never start. Window: 0 (11 records, none zero-length; 8 subs, all
  started). Under R3-2 such a sub gets NO id; its zero-length records keep the pool key (`direction` +
  `starting_idx`) with `sub_id` empty [R4-4].
- **The two read rules decide nothing on the window** (R4-5, R4-6): no x−1 threshold event (BOS / CTS confirmed,
  threshold updated, CTS established / updated) falls on any x's start candle; the only x−1 already ended at x's start
  is sub 3 (ended 3611) before sub 4 (3621) — a first_confluence sub, so the Q3 rule, not x−1.
- **For the plan session:** `trigger_sub_sid` (the per-`(lens, S, C)` record counter, taken at fire) numbers RECORDS —
  trigger logs, fire-ordered by nature — so R3-2 leaves it alone; the audit confirms nothing reads it as a start
  order.

**Round 5 (2026-10-07; the code + S_F1):**

- **The one ordering case the user's argument does not cover crosses a MAIN reversal** (the parent-floor lag, round
  4). A reversal structure's cycles can be established BEFORE it starts (the probe builds it back to its starting
  candle), and the cycle current at its start can stay live ACROSS the start: sub 2 (an M15 reversal structure, same
  MS) started 2470 with its cycle 0 established 2368 and its next at 2608 — no cycle established AT its start
  (subs 1 and 3 did establish one at their start: 1940, 2829). UNVERIFIED FOR H1 — a main reversal builds its
  structure through a different path from the pool's sub reversal; the plan session's audit checks whether an H1
  reversal structure's cycle can be live across its start. If it can: a trigger on such a cycle fired before the
  reversal starts at the reversal (its floor), while the OLD structure's last cycle is live until the reversal and
  can still create and start subs in between → fire order ≠ start order among subs that START. Window: not reachable —
  H1 sid 1's cycle 2 was established AT the reversal (H1 902 = M15 3611), so cycles 0 / 1 ended there and their 4
  triggers (2815 – 3487) were dropped; had cycle 1 stayed live, FC (1,1)@2995 would have been numbered at 2995 and
  started at 3611, after any sub sid 0's (0,1) created and started in between (none on the window: the 2995
  first_counter joined sub 3). In real time sid 1 does not exist before 3611, so R3-2 (number at the start) gives the
  real-time order — the user: "your fix also doesn't change things and is more correct".
- **The ending thresholds are the FROZEN ones** (the user's R4-6 addition) — reading x−1 by MOMENT up to its end gives
  exactly that: every reversal-ended sub's last `bos_threshold` == its applied reversal's `bos_frozen` (4 / 4: sub 0
  0.57877 @1940, sub 1 0.57926 @2470 — after watches at 0.57884 and 0.57897 whose failed breaks moved it — sub 2
  0.58156 @2829, sub 6 0.57898 @4200); sub 3 (ended 3611 by `parent_end`) had cycle 3 established 3589 but
  unconfirmed, so its same-direction read is cycle 2's `cts_threshold` 0.57376 (@3521) — the frozen threshold cycle 3's
  breakout broke; a pattern or reversal watch still pending at the end → the value the freeze holds (the later apply
  or failure comes after the end and never counts). The exported sub events are already clipped at the sub's end (0
  post-end sid-0 events on all 7 ended subs); the internal geometry is not — the plan reads through the clip.

## 4. Naming (ACCEPTED 2026-10-07, round 2 — "Everything else looks ok"; the definitions below)

- **N-a — the three cross-fib-limit fields** (renamed `cross_fib_limit_*` in round 3, R3-3 — "cross" alone reads
  ambiguously; "limit", not "ceiling" — the code's word: a sell structure's limit is BELOW):
  - **`cross_fib_limit_price`** (float | None) — the reference price `P` of a structure (an H1 main sid or a unique sub),
    set ONCE at the structure's start (main: the reversal's apply; a sub: its `start_idx`) by the rule of its starting
    trigger (§1.2) and never changed [R2-4]. None = no reference → no cross on any cycle (every cycle single).
  - **`cross_fib_limit_source`** (str) — which rule produced `cross_fib_limit_price`: `none` (main start trading; a sub's
    first_confluence on parent cycle 0; no previous sub), `reversal_bos_frozen` (main or sub reversal),
    `parent_cts_threshold` (first_confluence on parent cycle ≥ 1), `prev_sub_cts_threshold` /
    `prev_sub_bos_threshold` (R2-1), `prev_sub_starting_price` (R2-2), `prev_sub_latest_cts` (R2-b), `prev_sub_no_cts` (R6-3 (vi)). WHAT IT SHOWS:
    the price alone is a bare number; the source names the rule that produced it, so a CSV row / hover reads "limit
    0.57827 — sub 6's `bos_threshold`" without re-deriving it (like an event's `via`). Window values: H1 sid 0 `none`;
    H1 sid 1 `reversal_bos_frozen` 0.5736; sub 0 `none`; subs 1 / 2 / 3 `reversal_bos_frozen` 0.57877 / 0.57926 /
    0.58156; sub 4 `parent_cts_threshold` 0.57376; subs 5 / 6 / 7 `prev_sub_bos_threshold` 0.58105 / 0.57112 /
    0.57827. (KEPT — R3-4, round 4: "keep".)
  - **`cross_fib_limit_cycle`** (int | None) — `M`: the first cycle whose latest CTS anchor is STRICTLY beyond
    `cross_fib_limit_price` = the LAST cycle a cross may target; every later cycle is single. None while no cycle has
    gone beyond yet (a cross may target any cycle, subject to the dead-cycle walk) — and always None when there is no
    `P`. Set in real time, at the moment (a `CTS_ESTABLISHED` / `CTS_UPDATED`) a cycle's anchor first passes `P`;
    never changes after (a CTS anchor never regresses, and every cycle is confirmed before the next is established,
    §3.2) — so the limit never revokes a cross already made. [R6-6: the structure's M is set in real time; the copy
    exported on each fib record is its FINAL value, like `end_idx` — a record's "could a cross target this cycle?" is
    `cycle ≤ M` (or M None), already knowable when the record was decided, since only the cycles below it count.]
  - Where they live (for the plan session): on the structure (the pooled sub / the main sid's tracker state) and
    copied onto every fib record's meta (CSV + hover) — new, additive meta keys. [Session 2, §9.3 D1: P / source /
    from on `PooledStructure.cross_fib_limit` and the tracker's limits; M on the tracker; exported only through the
    fib records (no structure-level CSV); the hover line = R6-7.] Included (R6-3 (ii)):
    `cross_fib_limit_from_sub_id` (the sub that supplied `P` — x−1, or the reversing sub — by its final, start-numbered
    id, §9.3 D2).
- **N-b "beyond"** = strictly past a price in the structure's direction (the user's "exceeds (>)"); retire "clears"
  (the ceiling's inclusive word).
- **N-c `started_by` on main structures too** (`start_trading` | `reversal`), mirroring the sub's `started_by`
  (`first_confluence` | `first_counter` | `subsequent_confluence` | `subsequent_counter` | `reversal`) — the
  trigger-attached variants (I1).
- **N-d retire "Scenario 1 / 2 / 3"** (I1): Scenario 2 = a cross (from cycle 0), Scenario 3 = a single, Scenario 1 →
  Q4. **"single"** everywhere — the code also says `"intra"` (`select_fib_anchor_for_cycle`). KEEP A RECORD [R2-3]:
  the retired rules move verbatim into a "Retired logic (history)" appendix of `CROSS_CYCLE_FIB_SPEC.md` at the build.
- **N-e "pre-established cross"** (the code / FIB_LIFECYCLE_SPEC §6 name) for what the session called the "early
  cross".
- **N-f "latest CTS anchor"** of a cycle = the confirmed anchor, else the last `CTS_UPDATED` (§1.1).

## 5. Open questions

### Round 8 (2026-10-08, session 4 — the F14 random-tail search, §9.2 F16 / F17, §9.10) — ANSWERED 2026-10-08 (§6 R8-1, R8-2)

- **R8-1 What to do with F16** (MS confirms a CTS with an anchor after its confirmation candle; emission order ≠
  `processing_order_key` order; today's downstream draws the fib at the earlier anchor). The S2 tripwire's own shape
  (F14 (1)) — 0 hits. Candidates: (a) **park it to S5** — record it (here + the register), build S3 → S3b → S4 as
  planned (none of them touches MS or the event order; the latent divergence is the same before and after), and open
  it as the first round of the S5 session (an MS fix vs which order wins), where D5's tripwire would raise on it;
  (b) **fix MS before S3** — its own plan (e.g., a pattern applied inside an already back-filled range never confirms a
  CTS anchored after its apply candle); the revamp waits; (c) **decide now: emission order canonical** for the fib
  consumers (S5's at-emit tracker becomes the reference; the downstream's dispatch invariant, LANDMINES "Event Sort
  Order Is a Dispatch Invariant", changes); (d) **decide now: processing order canonical** (D5 as written) and S5's
  tripwire downgraded to a WARNING for this shape. Recommended: (a).
- **R8-2 What to do with F17** (the downstream silently skips a CTS_ESTABLISHED whose BOS anchor lies after its CTS
  anchor; the F14 (1) shape exists — 6 / 60,000 synthetic, 0 real, 0 window — and is hidden from the S2 tripwire this
  way). Candidates: (a) **park it with F16 to S5** — build S3 → S3b → S4; the skip stays silent; (b) **make the skip
  loud now** — the dispatch raises a named `RuntimeError` (`FibEventOrderError`) on a CTS_ESTABLISHED whose cycle's
  BOS_CONFIRMED was not handled first, instead of dropping it: behaviour-neutral on the window (0 / 34), R6-4's rule
  ("the order tripwire raises so the shape is seen"), its own small commit + byte-identical `/compare` before S3; the
  MS side (a cycle established on the previous cycle's CTS candle; a BOS after its CTS) goes to the S5 round with F16;
  (c) **fix MS before S3** — its own plan (a new cycle's CTS strictly beyond the previous cycle's / its BOS before its
  CTS); the revamp waits; (d) **re-order the dispatch** — hold a CTS_ESTABLISHED until its BOS_CONFIRMED arrives
  (changes the pinned processing order, LANDMINES "Event Sort Order Is a Dispatch Invariant"). Recommended: (b).

### Round 7 (2026-10-07, session 2 — four follow-ups to round 6) — ANSWERED 2026-10-07 (§6 R7-1 … R7-4)

- **R7-1** R6-1b was a sub-item of R6-1 — does R6-1's "ok" cover it (fix both latent single quirks in S3b)?
- **R7-2** R6-3 (iv) and (ix) had no recommendation: (iv) `fib_mode: "cross_cycle"` on every record says nothing after
  S4 — retire it after the build in its own small commit (the fib CSV loses its `fib_mode` column)?; (ix) a None
  `sub_id` prints its column as float (`3.0`), as `parent_bos_anchor_idx` already does — accept (0 on the window)?
- **R7-3** the label value: `past_cross_fib_limit` misreads on a structure with no `P` (sub 0, H1 sid 0 — no limit to be
  past). Candidates: keep it; one neutral value `cross_fib_limit` ("single because of the limit rule" — the source and
  cycle fields say which case); two values `past_cross_fib_limit` / `no_cross_fib_price`. Recommended: the neutral one.
- **R7-4** is PLAN_I §9 approved for the build (round 6 answered the questions, not the plan as a whole)?

### Round 6 (2026-10-07, session 2 — raised by the code audit and the cold review, §9.2 / §9.8) — ANSWERED 2026-10-07 (§6 R6-1 … R6-7: every recommendation taken)

- **R6-1 Which "single" for a cycle past the limit (and every cycle ≥ 1 of a structure without `P`)?** Two behaviours
  the sub logic has (§9.2 F4; main's plain single without late activation is already ruled out by I-lean / I5): (a)
  the sub's cycle-0 single — activates at CTS_ESTABLISHED when its own `[BOS, CTS]` holds an unfilled sd imbalance,
  else at the first CTS_UPDATED that finds one; every update re-checks it (deactivate / reactivate); no label; (c) the
  sub's `cross_failed` single — re-checked only while its own window is unfilled, label `via: cross_failed`; or (d) =
  (a) plus its own label, e.g. `via: "past_cross_fib_limit"` (a CSV row then says WHY it is single, as
  `cross_fib_limit_source` says why P is what it is — under (a) a reader must compare `cycle_id` with
  `cross_fib_limit_cycle`). Example: sub 1 cycle 3 (BOS 2078, 12-16 13:30 → CTS 2231, 12-18 03:45; sub 1's `P`
  0.57877, M = 2 → past the limit): today a `cross_failed` single; under (a) the same fib (start 2231, end 2343 12-19
  07:45) without the label, under (d) labelled. On the window (a) and (c) differ ONLY in that label (V3 vs V3c: `via`
  on 9 confluence + 2 counter rows). MY RECOMMENDATION: (d) — (a)'s behaviour, the one single the sub logic already
  has as a primary, with a label that keeps the CSV self-explaining.
  - **R6-1b two latent quirks of the single, both 0 on the window — fix or keep?** (i) the stale `cross_failed`
    single (F5): when cycle T's own window `[BOS_T, CTS]` is filled at a CTS_UPDATED, today's fallback single is not
    touched — it stays active on its older CTS, with every imbalance in its window filled; (ii) a late-activated single
    ends the PREVIOUS cycle late (F15): if cycle T's single activates at a CTS_UPDATED u after T's establishment e,
    cycle T−1's fib ends at u instead of e. Your R0 keeps "when & conditions that they are active … unfilled
    imbalances" unchanged inside the limit, so both are asked, not assumed. MY RECOMMENDATION: fix both as latent bugs
    in their own stage S3b (one single behaviour; a fib active with all its imbalances filled is a state no other path
    allows; a cycle's fib cannot outlive its cycle) — 0 cells on the window, pinned by constructed tests.
- **R6-2 When is a sub's `P` fixed — at its first trigger's FIRE (when its MS runs) or at its START?** A sub's MS run
  (where MS's in-flight POI read happens, Q7) is built at its first trigger's FIRE (the sweep's phase 0), before the sub
  has started; §1.2 reads `P` at the sub's start from its started_by record. Candidates: (a) compute `P` at the fire,
  from the creating trigger at its record's start (known at the fire), x−1 = the latest sub known to start by then —
  one `P` for the MS run and the drawn fib; after the sweep re-derive it the §1.2 way and log a WARNING on a mismatch
  (this knowingly departs from §1.2 in the mismatch cases); (b) as (a) but RAISE on a mismatch; (c) re-run the sub's MS
  with the re-derived `P` (can change the sub's structure → the sweep → other subs; not guaranteed to settle). Where
  (a) and §1.2 differ — all unreachable on the window: (1) a floor-lagged trigger: suppose H1 sid 1's cycle 1 had
  stayed live past the reversal at 902 (F1) — the first_counter fired on it at 3283 (01-06 02:45) would start at the
  floor 3611 (01-09 12:45); a sub y on sid 0's last cycle that fires AND starts at, say, 3400 (01-07 08:00) would then
  be the literal x−1, but at 3283 it does not exist yet, so (a) reads sub 3 instead; (2) the first-fired record is not the first-started
  one (it is zero-length because its parent cycle ended first, or a later-fired record of the same sub starts
  earlier); (3) a sub "known to start" at the fire never starts (frozen zero-length by a same-candle collision,
  `lifecycle_sweep.py:434–448`); (4) a lagged x whose x−1 ENDS inside (fire, start] — its end is unknown at the fire,
  so post-end events would be read. Window: the creating record == the started_by record 8 / 8; the 2 lagged starts
  (subs 0, 4) are first_confluence, whose `P` reads no previous sub. MY RECOMMENDATION: (a).
- **R6-3 Small choices (the plan's AUTHOR'S RESOLUTIONS — confirm or change any):** (ii) the optional
  `cross_fib_limit_from_sub_id` included (§4 left it to the plan); (iii) main's `started_by` (values decided, N-c) goes
  on its fib records and the main SidRecord's meta (no main structure table is exported); (iv) `fib_mode:
  "cross_cycle"` stays on every record (Q8) though no other mode remains — retire or rename it later?; (v) the
  `disappeared` status + its chart / POI filters stay as a generic hook with no producer (`scenario1_revert` retires);
  (vi) a same-direction x−1 with no CTS anchor at all by x's start → no `P`, a new source value `prev_sub_no_cts`, a
  WARNING (the rule does not cover it); (vii) the sub path's private `_m15_*` names become `_cross_check`, `_phase`, …
  in S4 (they serve H1 too); (viii) `sub_id` numbering by one pass at the sweep's end, sorted by (start, creation)
  — R3-2's own second option, not a phase-2 counter; (ix) a None `sub_id` cell makes pandas print the column as float
  (`3.0`), as `parent_bos_anchor_idx` already does — accept, or switch these columns to the `Int64` dtype?
- **R6-4 Q7's consequences — stated, not asked; one real question.** MS reading the drawn fib exactly also removes the
  M1 divergence (MS counting a gap whose third candle has not formed — your Plan F rule) and MS reading a fib
  FibTracker never creates or has deactivated; both are implied by Q7 (window: 0 cells — the inner differs at 20 / 371
  refreshes, e.g. sub 7 cycle 2 at 4179 01-19 10:45, 0 / 475 proximity answers change). The open part is the ORDER
  edge (§9.2 F14): the drawn fib processes a cycle's CTS_ESTABLISHED at its anchor, MS at its moment; they disagree
  only if another of that structure's events falls in between — e.g. cycle T−1 confirmed by proximity during the
  candles of cycle T's breakout pattern, which also mis-handles TODAY's drawn fib (cycle T−1 missing from the walk,
  cycle T's phase reset). 0 / 34 on the window. Candidates for the tripwire: (a) RAISE (a named error — the replay
  stops, the shape becomes a new round); (b) a WARNING and carry on. MY RECOMMENDATION: (a) — the case is unmeasured
  and silently wrong today; a stop is how it gets seen.
- **R6-5 S4 as ONE stage?** S4 carries seven of your decisions, each with its own delta on the window (§9.5 c): Q6 → +
  (0,0) single 96 → 430 (drawn, no POI); Q4 → + (1,0) single, collapsed; I2 → (1,1)'s cross pre-established at 741
  (12-30 19:00) instead of 748; I5 → (1,2) becomes a `cross_failed` single (same anchors); Q8 → the format of (0,1),
  (1,1), (1,2); Q5 and main's P source → no delta. All of them live in the h1 handlers S4 deletes. Candidates: (a) one
  stage, its `/compare` checked cell by cell against these per-cause lines (they are disjoint rows / keys); (b) S4a =
  Q8's format on today's h1 code, S4b = the switch; (c) one stage per cause, each written onto the h1 code first.
  MY RECOMMENDATION: (a) — (b) and (c) write the sub's behaviour or keys onto code deleted days later, and every delta
  is already attributed to one cause by the variants.
- **R6-6 `cross_fib_limit_cycle` (M) — real-time or final?** §4 N-a (accepted) defines M as set in real time ("None while
  no cycle has gone beyond yet … set at the moment a cycle's anchor first passes P"). Copied onto each fib record,
  that leaves a choice. Example: sub 7 (P 0.57827) — M = 2 is set at 4179 (01-19 10:45), cycle 2's CTS_ESTABLISHED
  (anchor 0.57856, already beyond); its cycle 0 / 1 fib records were decided before that. Candidates: (a) the FINAL M on
  every record (sub 7's cycle 0 / 1 / 2 records all read 2) — a record's question "could a cross target this cycle?" is exactly `cycle ≤ M` (or M None),
  and that answer was knowable when the record was decided, because only cycles below it count; (b) M as of each
  record's own decision (sub 7's cycle 0 / 1 / 2 records read None / None / 2); (c) both — `cross_fib_limit_cycle` final + `cross_fib_limit_cycle_at_decision`. MY RECOMMENDATION: (a),
  with N-a amended to say the structure's M is set in real time and the exported copy is its final value (like
  `end_idx`).
- **R6-7 The hover (§4 N-a / R3-4 said "CSV + hover"; the plan had dropped it).** The fib hover today (both charts)
  shows status / life / IP. Proposed: one more line, e.g. sub 7 cycle 1 "limit 0.57827 · prev_sub_bos_threshold (sub 6)
  · M=2", H1 sid 1 "limit 0.5736 · reversal_bos_frozen · M=2", no P "limit: none (first_confluence on parent cycle
  0)". In S2 (format-only, with a rendered example for your review before it lands — your chart rule); or defer it
  (a register line). MY RECOMMENDATION: in S2, the text above, rendered first.

### Round 2 (asked 2026-10-07) — R2-b ANSWERED ((b), round 3); R2-a → R3-2 (ACCEPTED round 4)

- **R2-a "x−1" — by `sub_id` or by start order?** `sub_id` is assigned when the sub's FIRST trigger fires (the sweep's
  phase 0, `SubStructurePool.get_or_create`), but `start_idx` = max(probe finalize, trigger, parent floor) can lag far
  behind (conf sub 0: trigger 463, start 1020) — so a sub created earlier can START later than one created after it,
  and at x's start the literal x−1 may not be live yet. Candidates: (i) literally `sub_id` x−1; (ii) the most
  recently STARTED unique sub before x (the latest `start_idx` ≤ x's; a tie → the lower `sub_id` is the earlier);
  (iii) as (ii) but `start_idx` strictly < x's (same-candle starts excluded). No such sub → no `P` (all readings).
  Window: identical (id order == start order, 8 / 8). MY RECOMMENDATION: (ii) — the user's "in live realtime, at the
  start_idx (moment)".
  **Round 3:** the user prefers (i) — "at each new unique sub x’s start_idx, sub x-1 should have started" — and asked
  how a previous `sub_id` could start later. Answer + measurement §3.4: ids are taken at the first trigger's FIRE (the
  offline probe), starts at the probe's finalize; they disagree only when a new sub fires and starts inside a
  first_confluence sub's fire → start gap (never on the window). → R3-2: number subs at their START so (i) holds by
  construction. **ACCEPTED round 4 (R4-2).**
- **R2-b Same direction, x−1 has no CONFIRMED CTS yet at x's start** (MS's `cts_threshold` exists only from a cycle's
  confirmation: None from cycle 0's establishment until its `CTS_CONFIRMED`). Real on the window's timeline, not hit:
  sub 4 started 3621 and first confirmed 3741 (01-12 21:15); sub 6 started 3819 and first confirmed 4157 (01-19
  05:15) — a same-direction sub starting inside either gap would have no `cts_threshold` to read. Candidates: (a) no
  `P` → no cross (the strict reading of "do not use CTS anchors … without having been confirmed"); (b) x−1's LATEST
  CTS anchor price at x's start (established / updated) — the analogue of R2-2 (a structural price of x−1 on the
  right side when its threshold does not exist yet); (c) walk back to the next-earlier sub and apply the same rule to
  it; (d) x−1's starting price — the WRONG side (a sell x−1 started at a high, above a sell x) → every cycle is beyond
  → `M = 0` → the same as (a). MY RECOMMENDATION: (b). **ANSWERED round 3: (b)** — "X-1’s latest as fallback is ok."

### Round 1 — ALL ANSWERED 2026-10-07 (§6; kept as asked, for the record)

- **Q1** "The previous unique sub" (first_counter / subsequent_confluence / subsequent_counter) — which sub, and as of
  when? Readings: (a) the most recent OPPOSITE-direction sub started at or before the trigger (the structure this
  move runs against — the reversal analogue); (b) the previous sub on the trigger's lens; (c) the sub it replaces;
  (d) the live sub on the OTHER lens. Window (§3.1): sub 7 (a)/(b) `P` 0.57827 → `M = 2` (its crosses kept) vs (c)
  0.57112 → `M = 0` (both lost — a same-direction sub's BOS is on the wrong side, so `M = 0` always); sub 6 differs
  between (a) and (b)/(c) but has one cycle. As of: the sub's `bos_threshold` at the trigger candle (knowable then). No
  such sub → no `P` (no cross)? MY RECOMMENDATION: (a), at the trigger candle; none → no cross.
- **Q2** "Exceeds (>)" strict vs today's inclusive `>=`: conf sub 1 cyc 1 (CTS 1898 = `P` 0.57877, the probe) → strict
  `M = 2`, inclusive `M = 1` (no output difference on the window). MY RECOMMENDATION: strict — a probe that did not
  break the barrier is not the break.
- **Q3** First_confluence `P`: (i) parent cycle c−1's `cts_threshold` at (S, c)'s `CTS_ESTABLISHED` (sub 4: 0.57376,
  after the range sync at H1 880), or (ii) parent cycle c−1's confirmed CTS anchor price (0.574). MY RECOMMENDATION:
  (i) — the threshold the pattern actually broke, as for the reversal. (No window output difference: one cycle.)
- **Q4** (I3) The Scenario-1 / revert gate — remove, the limit replaces it? Today, main sid ≥ 1 only: Scenario 1 =
  CTS_0's moment ≥ the reversal's apply (a TIMING test) → cycle 0 gets a single fib, cycle 1+ single; else no cycle-0
  fib and cycle 1 decides cross vs single; the revert = BOS_1 reaching back into the previous sid's last BOS zone →
  the cycle-0 fib "disappears" and cycle 1 may cross. Under the limit: cycle 0 is single whenever no cross is allowed
  (a sub's cycle 0 today), `M = 0` when CTS_0 is beyond `P` (a PRICE test) — conf sub 2 shows they differ (§3.1). The
  revert has no analogue (a revert after `M = 0` would keep cycle 1 single). Not exercised on the window's main (sid 1:
  Scenario 1 FALSE). MY RECOMMENDATION: remove both.
- **Q5** (I4) Fill-as-of: main's cycle-1 cross judges cycle 0's liveness at frozen points (at CTS_0 and at BOS_1) —
  a SUPERSET of the sub's "current candle" check (it keeps a cross alive after cycle 0's gaps fill in BOS_1 → CTS_1).
  MY RECOMMENDATION: the sub's "current" everywhere (one routine; already main's rule for targets ≥ 2). Window: no
  difference — sid 1 cyc 0 [689, 710] still holds an unfilled sell gap at every candle 748 → 902 (re-measured §3.1),
  so the current check passes wherever the snapshot does.
- **Q6** Main start-trading cycle 0: today NO fib (`fib_tracker` "sid=0 cycle=0 NO FIB (simple flow)"), while a sub's
  cycle 0 — sub 0's, started on parent (0,0) with no `P` either — gets a single. One logic → H1 sid 0 cyc 0 gets a
  single fib (BOS 96 → CTS 430, 11-20 → 12-10)? It adds an H1 fib + an IP POI record (and through MS's sd-proximity,
  possibly (0,0)'s confirmation — today by pullback at 439). Open.
- **Q7** (I7) MS's in-flight POI read: MS confirms a CTS by sd-proximity to the cycle's POI inner (e.g. H1 (1,2)
  `CTS_CONFIRMED` 939 `sd_zone_proximity`, inner 0.57866), so it computes that POI DURING its run, from its own copy
  of the fib decision: on a sub always the cycle's single fib, on main a cross only for cycle 1. The drawn POI comes
  from FibTracker's fib. Example (PLAN_H §3, D5): conf sub 0 cyc 2 — drawn fib 454 → 1245 (a cross), MS's fib 1107 →
  1245 (single), zones ~100 pips apart. MY RECOMMENDATION: MS reads the SAME fib decision (the shared routine with the
  limit), so MS confirms on the drawn POI — it needs `P` inside every MS run (main: its own reversal's `bos_frozen`;
  a sub: handed in from the pool).
- **Q8** (I6) "Bespoke" = made one-off, its own format: main's cross fib record carries its own meta
  (`cross_cycle`, `scenario: 2`, `cycle1_bos_anchor_idx`) and NOT the sub's (`version`, `fib_mode`,
  `cross_start_cycle`) — so its fib CSV row has empty `version` / `fib_mode` cells (H1 sid 1 cyc 1 in S_F1). MY
  RECOMMENDATION: main adopts the sub's meta (falls out of one code path); a meta change → the rule-3 atomic migration.

## 6. Decisions

**2026-10-07 (round 1 — the user's spec; `plan_i_inputs/cross_cycle_fib_spec.md`)**

| ID | Decision | The user's words |
|---|---|---|
| R0 | The cross limit: cross fibs from cycle 0 up to the first cycle whose CTS anchor exceeds the reference price in the structure's direction; every later cycle single; main AND sub; within the limit every other cross rule unchanged | "cross-cycle fibs can form starting from cycle 0 up to the first cycle whose CTS_anchor exceeds (>) a reference price / threshold in the structure’s direction … afterwards any additional cycles will have single fibs … all other logic … should all still remain the same" |
| R0a | The reference is set by the structure's starting trigger (§1.2); a unique sub's by its FIRST trigger in real time | "the unique sub’s reference threshold price for cross fibs will be set depending on the very first trigger (in actual realtime) mapped / assigned to the unique sub" |
| I1 | Variants attach to triggers, not sid; retire "Scenario X" names | "Rather than applying logic based on sid. We now attach variations to triggers. Going forward, we should move away from using the Scenario X terminology" |
| I2 | The pre-established cross runs on main too, within the limit | "Early cross: I think this we can actually unify and have on main as well … early cross should be allowed within the confines that a cross-cycle fib would be allowed after reversal." |
| I5 | Post-reversal fib handling on main = the sub's after a reversal-started sub | "how we deal with fibs after unique sub first triggered via reversal should be same as how we deal with fibs post reversal on main." |
| I8 | No single + cross coexistence | "We shouldn’t have coexistence with the updated logic" |
| I-lean | Where main and sub differ (I4, I7), unify on the sub's logic | "I’m inclined on 4 and 7, where we may have differences between sub and main to unify based on what we have in sub. Even without our current edits, the sub logic is already more advanced." |

**2026-10-07 (round 1 — the answers to Q1–Q8)**

| ID | Decision | The user's words |
|---|---|---|
| Q1 | first_counter / subsequent_confluence / subsequent_counter: `P` = the `bos_threshold`, at the trigger candle, of the MOST RECENT previous unique sub whose direction is OPPOSITE the starting sub's (reading (a)); none → no `P` (no cross). A unique sub's direction is fixed for its life (a reversal starts a NEW unique sub), so "its most recent direction / direction at end" = its direction (the RELATIVE direction to the parent — confluence / counter, `relative_dir_segments` — can change, and is not what is compared). Window: sub 5 ← sub 4 (0.58105), sub 6 ← sub 5 (0.57112), sub 7 ← sub 6 (0.57827 → `M = 2`, its crosses kept) | "It should be a), and again emphasis on it should be most recent previous unique sub whose most recent direrction or direction at end is opposite the direction of the current unique sub that is starting due to the first time trigger being one of these mentioned." (AskUserQuestion option (a): "none exists → no reference (no cross)") |
| Q2 | "Beyond" is STRICT (sd = +1: `>`; sd = −1: `<`); reaching the price is not beyond — today's inclusive `>=` / `<=` goes. Window: conf sub 1 `M = 2` (the breaking cycle), not 1 (the probe) | "Strict > (Recommended)" |
| Q3 | first_confluence on parent cycle c ≥ 1: `P` = the parent's cycle c−1 `cts_threshold` as of parent (S, c)'s `CTS_ESTABLISHED` (sub 4: 0.57376) | accepted as recommended ("Recommendations I'll take unless you object") — no objection |
| Q4 | REMOVE main's Scenario-1 gate and the revert; the limit replaces them (post-reversal cycle 0 = single, `M = 0` when CTS_0 is beyond `P`; no revert path) | "Remove both (Recommended)" |
| Q5 | Fill-as-of = the sub's "current candle" everywhere (main's cycle-1 frozen snapshot points go) | accepted as recommended — no objection |
| Q6 | Main's start-trading structure gets a cycle-0 SINGLE fib too (H1 sid 0 cyc 0, BOS 96 → CTS 430) — every structure's cycle 0 is single | "Yes, single fib" |
| Q7 | MS's in-flight POI read uses the SAME fib decision as the drawn POI (the shared routine, with the limit) — `P` inside every MS run (main: its own reversal's `bos_frozen`; a sub: handed in from the pool) | accepted as recommended — no objection |
| Q8 | Main adopts the sub's fib meta (`version`, `fib_mode`, `cross_start_cycle`; `scenario` / `cycle1_bos_anchor_idx` go) — a rule-3 atomic migration | accepted as recommended — no objection |

**2026-10-07 (round 2 — the user's feedback on the decided rules; verbatim in the spec copy, part 3)**

| ID | Decision | The user's words |
|---|---|---|
| R2-0 | §1.4 points 1 (cycle 0 single everywhere), 2 (the cross condition), 4 (no Scenario 1 / revert), 5 (one fib decision for the drawn fib + MS), 6 (one meta format) confirmed | "1. Sounds good all cycle 0 will have single fib 2. Sounds good … 4. Sounds good … 5. Ok 6. Ok" |
| R2-1 | first_counter / subsequent_confluence / subsequent_counter: `P` from the PREVIOUS unique sub x−1, read at x's `start_idx`: same direction → x−1's ending `cts_threshold` of its most recent CONFIRMED CTS (never an only-established / updated one); opposite → x−1's ending `bos_threshold` of its most recent BOS. SUPERSEDES Q1. Window: subs 5 / 6 / 7 all have an opposite x−1 → `P` unchanged (§3.3) | "In live realtime, at the start_idx (moment) of unique sub (sub_id=x), - If sub_id=x-1 most recent / last / ending direction is the same as sub_id=x, then reference price will be most recent / ending cts_threshold attached to the most recent confirmed CTS anchor (do not use CTS anchors that are only established or updated without having been confirmed) - If sub_id=x-1 most recent / last / ending direction is opposite of sub_id=x, then reference price will be most recent / ending bos_threshold attached to the most recent BOS anchor … The reference price should always be coming from the x-1 unique sub" |
| R2-2 | Opposite-direction x−1 with no `bos_threshold` yet → x−1's finalized `starting_idx` price (its starting candle's extreme on its origin side — the high of a sell sub, the low of a buy sub — = its BOS_0 price, 8 / 8) | "This edge case shouldn’t happen. In the updated logic, if previous sub is opposite direction with no bos_threshold, then we simply use the previous sub’s finalized starting_idx as fallback." |
| R2-3 | Retire Scenario 1 / 2 / 3 and the revert, but KEEP A RECORD: at the build, the retired rules move verbatim into a "Retired logic (history)" appendix of `CROSS_CYCLE_FIB_SPEC.md` (with the last commit that ran them); the GLOSSARY entries are marked retired with a pointer | "I would still like to have the logic documented somewhere for historical safekeeping and so we can reference the iterations we’ve gone through in the future if needed." · "yes retire Scenario 1/2/3, but still keep a record for the future" |
| R2-4 | `P` is determined ONCE at the structure's start moment and never changes | "the reference price is determined once at the unique sub’s start_idx moment, and then it does not change" |
| R2-5 | Fib activeness stays clamped by the lifecycle — unchanged (§1.4) | "this should already be implemented, but fib activeness should be clamped by the cycle’s lifecycle so that doesn’t change" |
| R2-N | The naming of §4 accepted (N-a … N-f) | "Everything else looks ok" |
| R2-Q3 | Q3 stands; the user asked whether MS has a `cts_frozen` mirroring the reversal's `bos_frozen` — no VARIABLE, but the same process (corrected in round 3, R3-1; §3.3) | "do we have some similar concept as applied breakout’s cts_frozen? If so that would mirror reversal. Just curious on this." |

The commit of these spec docs waits for the session's end (the user: "Let’s hold off for end of the session").

**2026-10-07 (round 3 — the user's detailed responses; verbatim in the spec copy, part 4)**

| ID | Decision | The user's words |
|---|---|---|
| R3-1 | The breakout (CTS-establishing) side MUST mirror the reversal: a pattern closes beyond `cts_threshold`; a failed break moves the threshold; the threshold is frozen while a pattern is pending. VERIFIED in today's code (§3.3, corrected): no `cts_frozen` variable, the same process. Q3's `P` = that frozen threshold; the plan session may stamp it on `CTS_ESTABLISHED` as `cts_frozen` | "Even if cts_frozen as a variable does not exist, the concept / process should be similar / mirror each other for both reversal and breakout." |
| R3-2 | ACCEPTED round 4 (R4-2): "x−1" = literally `sub_id` − 1, made real-time true by numbering each unique sub at its START, not at its first trigger's fire (§3.4) | "It should the literal sub_id - 1 … By unique sub's nature, their identifier incrementation should be something that aligns with realtime start." · "at each new unique sub x’s start_idx, sub x-1 should have started, which is why I feel i) should work. But I’m open to confirming & double checking." |
| R3-3 | Rename the fields `cross_fib_limit_price` / `cross_fib_limit_cycle` / `cross_fib_limit_source` | "Let’s use cross_fib_limit_xxx, so that it is more intuitive? Cross by itself sounds a bit confusing" |
| R3-4 | `cross_fib_limit_source` explained (§4) — KEPT round 4 (R4-3) | "I’m still not quite sure what the source values are and what this field is supposed to show?" |
| R3-5 | R2-b = (b): a same-direction x−1 without a confirmed CTS at x's start → x−1's LATEST CTS anchor price (established / updated) at that moment; source `prev_sub_latest_cts` | "X-1’s latest as fallback is ok." |
| R3-6 | Docs / replies name a unique sub WITHOUT a lens prefix ("sub 5"); the lens is where it is drawn ("sub 5, charted on counter") | "Are unique subs not independent of lens? … There should be only 1 unique sub 5, 1 unique 6, and 1 unique 7 correct?" |

**2026-10-07 (round 4 — the user's replies to round 3; verbatim in the spec copy, part 5)**

| ID | Decision | The user's words |
|---|---|---|
| R4-1 | R3-1 confirmed, incl. the plan-session stamp of the frozen threshold on `CTS_ESTABLISHED` (meta `cts_frozen`, read like `bos_frozen`; an additive meta key) | "Thank you on verifying. We are good here" |
| R4-2 | R3-2 ACCEPTED: unique subs are numbered at their lifecycle START (the sweep's phase 2), not at their first trigger's fire → "x−1" = `sub_id` − 1 has always started by x's start; assert `sub_id` order == `start_idx` order (a same-candle tie → creation order); contract-level (`sub_id` = the join key) → one atomic commit. A TriggerRecord's `start_idx` is never NULL (known at fire) | "On R3-2. yes I agree with this change. Unique sub numbering should be based on life cycle start." · "R3-2. Ok" |
| R4-3 | R3-4: keep `cross_fib_limit_source` | "R3-4. keep" |
| R4-4 | No numbered sub may never start (allowed today, §3.4 round 4): with R4-2 a sub that never starts gets no id; its zero-length records keep `direction` + `starting_idx`, with `sub_id` empty | "if it’s the case there are unique subs numbered that never starts, then that shouldn’t be happening even now." |
| R4-5 | ACCEPTED round 5 (R5-2): x−1 is read INCLUSIVE of x's start candle — its events whose moment ≤ x's `start_idx` count (the codebase's convention: the sibling read's `hi` is inclusive) | — |
| R4-6 | ACCEPTED round 5 (R5-3, + the frozen ending thresholds): an x−1 that has already ENDED at x's start is read as of its `end_idx` (its final thresholds); its geometry's post-end events (the MS runs on to the data edge) never count — the clip the sibling read already applies | the user's "most recent / last / ending" (R2-1) |
| R4-7 | WITHDRAWN round 5 (R5-1) — the claim was wrong: a first_counter cannot start before its parent cycle's first_confluence sub (§3.4, corrected). Was: "PARKED (register D.2, DECIDE): today a first_counter (or any sub) can start before its parent cycle's first_confluence sub has started" — the register line, added the same session and never committed, was removed | "by definition what followed first confluence would be first counter, and by the time first counter & the sub starts, it should always be the case that the first confluence sub has started" |

**2026-10-07 (round 5 — the user's replies to round 4; verbatim in the spec copy, part 6)**

| ID | Decision | The user's words |
|---|---|---|
| R5-1 | The ordering argument holds (verified, §3.4 round 4 corrected): within a parent structure, a sub that starts never starts before one that fired earlier — the first_counter fires only at / after the parent cycle's `CTS_CONFIRMED`, the first_confluence's probe is bounded by that cycle's confirmed CTS ANCHOR. R4-7 withdrawn. R3-2 still adopted: it numbers never-started subs not at all (R4-4) and orders the one cross-reversal case (§3.4 round 5) in real time; no change on the window | "by trigger & probe end logic, first counter subs can’t start before first confluence subs … This is all based all innately laid out in the logic so no explicit enforcement is necessary." · "I think your fix also doesn’t change things and is more correct so let’s go with that." |
| R5-2 | R4-5 accepted (x−1 read inclusive of x's start candle) | "R4-5. Yes this is important to keep" |
| R5-3 | R4-6 accepted, with the user's addition: an ended x−1's thresholds stop updating at its end; when it ended by a reversal or a new parent cycle, its ending thresholds are the FROZEN ones (a reversal → the applied `bos_frozen`; a pending pattern / watch → the held value). Verified: reading by moment up to the end yields them (§3.4 round 5: 4 / 4 reversal-ended subs) | "when I said “ending” it is for situations where the x-1 structure has ended, and at that point we would assume those bos or cts thresholds at that point stops updating … when the structure ended due to say new structure (reversal) or new parent (new parent cycle), the ending thresholds we should be using are the frozen ones we discussed." · "But yes both defaults are fine." |

**2026-10-07 (round 6 — session 2's code-raised questions, §5; verbatim in the spec copy, part 7 — every recommendation taken)**

| ID | Decision | The user's words |
|---|---|---|
| R6-1 | (d): every single the limit forces (cycle ≥ 1 with no `P`, or past `M`) behaves as the sub's cycle-0 single (activate at CTS_ESTABLISHED, else late on a CTS_UPDATED; every update re-checks) and carries a `via` label — its value `cross_fib_limit` (R7-3; R6-1 had offered `past_cross_fib_limit`) | "R6-1. ok sounds good" |
| R6-1b | Fix both latent quirks of the single in their own stage S3b: the stale `cross_failed` single re-checks its own window on every update (F5); a late-activated single ends its previous cycle at its own cycle's establishment, not at the late activation (F15). 0 cells on the window | taken with R6-1's "ok" (the sub-question was listed under R6-1) — CONFIRMED, R7-1 |
| R6-2 | (a): a sub's `P` is computed at its first trigger's fire (when its MS runs), from that trigger at its record's start; after the sweep it is re-derived from the started_by record at the sub's start with the start-numbered x−1, a WARNING on any mismatch, the stored `P` kept (a known departure from §1.2 in the four cases R6-2 lists; 0 on the window) | "R6-2. ok" |
| R6-3 | The author's resolutions: (ii) `cross_fib_limit_from_sub_id` included; (iii) main's `started_by` on its fib records + the main SidRecord's meta; (iv) `fib_mode: "cross_cycle"` kept on every record through the build, then retired (R7-2); (v) the `disappeared` status kept as a hook with no producer; (vi) no CTS at all on a same-direction x−1 → no `P`, source `prev_sub_no_cts`, a WARNING; (vii) the `_m15_*` names drop `m15_` in S4; (viii) `sub_id` numbered in one pass at the sweep's end by (start, creation); (ix) the float rendering of a None `sub_id` column accepted (R7-2) | "R6-3. your recs sounds good" |
| R6-4 | Q7's consequences stated (the M1 divergence and the never-created / deactivated fib go with it); the order tripwire RAISES a named error (`FibEventOrderError`) — the replay stops and the shape becomes a new round | "R6-4. your rec sounds good" |
| R6-5 | S4 is ONE stage; its `/compare` is checked cell by cell against the per-cause lines of §9.5 c | "R6-5. yes" |
| R6-6 | (a): the exported `cross_fib_limit_cycle` is the structure's FINAL `M` on every fib record (§4 N-a amended) | "R6-6. your rec sounds good" |
| R6-7 | The hover gains one line in S2 (e.g. "limit 0.57827 · prev_sub_bos_threshold (sub 6) · M=2"), rendered for the user's review before S2 lands. **Rendered 2026-10-08** (the last line of the status cell, below PLAN_H's `IP <rank> · IC <idx>` line; 24 drawn fib hovers) and **APPROVED as is** — `.5f` price, the order kept (§9.10 S2; spec copy part 9) | "R6-7. ok"; 2026-10-08: "(a) approve as is, commit S2" |

**2026-10-07 (round 7 — the four follow-ups, §5; verbatim in the spec copy, part 8)**

| ID | Decision | The user's words |
|---|---|---|
| R7-1 | R6-1b confirmed: both latent single quirks fixed in S3b | "1. yes" |
| R7-2 | (iv) the `fib_mode` meta key is retired AFTER the build in its own rule-3 commit (S6, §9.4); (ix) the float rendering of a None `sub_id` column is accepted | "2. sounds good on both" |
| R7-3 | The label of a single the limit forces is `via: "cross_fib_limit"` (one neutral value; `cross_fib_limit_source` / `_cycle` say which case) | "3. neutral sounds good" |
| R7-4 | PLAN_I §9 APPROVED for the build (session 3 starts with S1 + S2) | "4. sounds good." |

**2026-10-08 (round 8 — the session-4 F14 random-tail search, §5; verbatim in the spec copy, part 10)**

| ID | Decision | The user's words |
|---|---|---|
| R8-1 | (a): F16 is PARKED to the S5 session — recorded here + the register; S3 → S3b → S4 built as planned (none touches MS or the event order); F16 opens the S5 session's first round (an MS fix vs which order wins) | "(a) Park to S5 (Recommended)" |
| R8-2 | (b): the downstream dispatch RAISES `FibEventOrderError` on a CTS_ESTABLISHED whose cycle's BOS_CONFIRMED was not handled first (it dropped it silently) — stage S2b, its own commit + byte-identical `/compare` before S3 (§9.4); the MS side of F17 (a cycle established on the previous cycle's CTS candle; a BOS after its CTS) goes to the S5 round with F16 | "(b) Make the skip loud (Recommended)" |

**Corrected reading (round 1, for the record):** the user's I1 note — "in order for the first confluence of parent
cycle 0 to trigger, the reversal needs to have happened first. Thus the unique sub for any parent cycle id 0 first
confluence trigger will not be that unique sub’s first trigger" — no M15 trigger comes from an H1 (main) reversal: a
sub `reversal` trigger is spawned only when an M15 SUB reverses (`lifecycle_sweep._phase0_spawn`). A first_confluence
on parent (sid ≥ 1, cycle 0) CAN be a unique sub's first trigger, when that parent cycle is live; on the window
(1,0) and (1,1) are degenerate (they end at the reversal's apply 902, their floor) → their 4 triggers are UNRESOLVED.
The rule ("parent cycle 0 → no reference") covers the case either way.

## 7. Change log

- 2026-10-07 — opened: the user's spec copied verbatim; round-1 reading (§1), the census (§3), naming proposals (§4),
  questions Q1–Q8 (§5), decisions R0 / R0a / I1 / I2 / I5 / I8 (§6).
- 2026-10-07 — round 1 answered (§6 Q1–Q8): (a) the opposite-direction previous sub; strict "beyond"; Scenario 1 +
  revert removed; cycle 0 single everywhere (incl. main start trading); Q3 / Q5 / Q7 / Q8 as recommended. §1.4 = the
  rule as decided; the spec copy amended in place with the decision IDs.
- 2026-10-07 — round 2 answered (§6 R2-0 … R2-5, R2-N, R2-Q3): the previous-sub rule (x−1, same → confirmed
  `cts_threshold`, opposite → `bos_threshold`; supersedes Q1), its opposite-direction fallback, `P` fixed at the start,
  the clamp unchanged, Scenario names retired with a history appendix, the naming accepted (§4 definitions); measured
  §3.3 (no window change; no frozen CTS barrier in MS). Open: §5 R2-a, R2-b.
- 2026-10-07 — round 3 answered (§6 R3-1 … R3-6): the CTS-side freeze mirrors the reversal (round 2's §3.3 statement
  corrected in place — every breakout must close beyond the threshold; frozen anchor → apply; a failed break moves
  it); R2-b = (b); fields renamed `cross_fib_limit_*`; the source field explained (R3-4 open); `sub_id` timing measured
  (§3.4) → R3-2 open (number subs at their start); lens-free sub labels.
- 2026-10-07 — round 4 answered (§6 R4-1 … R4-7): R3-2 accepted (subs numbered at their start), the source field
  kept, no numbered sub may never start (R4-4); two read rules stated as defaults (R4-5 inclusive, R4-6 an ended
  x−1); §3.4 corrected in place (a lag has TWO sources — the FC finalize and the parent floor) + the round-4 findings
  (a first_counter can start before its FC sibling — parked, R4-7).
- 2026-10-07 — round 5 answered (§6 R5-1 … R5-3): the user's ordering argument verified (a first_counter cannot start
  before its FC sibling — round 4's claim corrected in place, R4-7 withdrawn, its register line removed); the one
  cross-reversal ordering case recorded (§3.4 round 5); R4-5 / R4-6 accepted, an ended x−1 yields its frozen
  thresholds (verified 4 / 4). Spec agreed; no open question.
- 2026-10-07 — session 1 CLOSED: §8 = the handoff to session 2; docs-only commit (local, not pushed).
- 2026-10-07 — session 2: the fresh code audit + variant replays V0–V5 on S_F1 (`plan_i_inputs/variant_patch_i.py`
  and friends; V0 reproduced S_F1 27 / 27) → §9 (the plan: stages S1–S5, every prediction cell-level, the test census);
  §2 and §3.1 corrected in place (Q6's (0,0) fib has no POI and cannot move (0,0)'s confirmation; the pre-established
  cross fires on main once, at 741); round 6 opened (§5, R6-1 … R6-5).
- 2026-10-07 — session 2: the cold review (3 read-only lenses, 930,594 tokens; 0 BLOCKER, 22 MAJOR, 38 MINOR across
  the lenses, §9.8) folded: §9 rewritten (the audit table +18 sites; F8 / F12 restated as measurements; F14 the
  anchor-vs-moment order edge — latent today; F15 the late-activation end; D1–D5 pinned: `pool_idx` kept out of the
  CSVs, the geometry signature chain, the final-id remap, the named errors, the tripwires, PR1, required arguments;
  chart predictions measured per stage; the census + its static guards); round 6 restructured (R6-1 (b) dropped as
  already decided, R6-4 a statement + the order question, R6-5 per cause, R6-6 real-time vs final M, R6-7 the hover);
  §1.2 / §1.4 / §4 amended in place.
- 2026-10-07 — session 2: round 6 ANSWERED (§6 R6-1 … R6-7, every recommendation taken; the user's words in the spec
  copy, part 7): R6-1 (d) the past-limit single labelled `past_cross_fib_limit` (§9.5 b / c updated), S3b in, R6-2 (a),
  R6-4 the order tripwire raises, R6-5 S4 one stage, R6-6 the final M (§4 amended), R6-7 the hover in S2. The census
  lists persisted to `plan_i_inputs/census/`.
- 2026-10-07 — session 2: round 7 ANSWERED (§6 R7-1 … R7-4; the spec copy, part 8): R6-1b confirmed (S3b), `fib_mode`
  retired after the build (S6), the float `sub_id` rendering accepted, the label `via: "cross_fib_limit"`; **the plan
  APPROVED** — session 3 = the build (S1 + S2 first).
- 2026-10-07 — session 3: **S1 LANDED** (§9.10): `sub_id` numbered at the lifecycle start, byte-identical to S_F1;
  2-lens landing review folded; one latent sweep quirk parked (register D.2).
- 2026-10-08 — session 3: **S2 LANDED** (§9.10): the cross-fib limit fields stamped, not applied — format-only = §9.5 a
  (27 / 27 CSVs == V2; 24 fib hovers + the limit line); the hover APPROVED as rendered (§6 R6-7; spec copy part 9);
  the landing review's M1 → D2 / §1.2 amended (main's cross-check reads the OPEN WATCH, not the applied candidate).
- 2026-10-08 — session 4: the F14 random-tail search (§9.10): the S2 tripwire's shape 0 hits; a new MS mechanism
  found (§9.2 F16 — a CTS confirmed with an anchor after its confirmation candle; emission ≠ processing order) →
  round 8 opened (§5 R8-1). R8-1 answered (a): F16 parked to the S5 session (register D.1 item 2). A 60k WAVES run
  then found the F14 (1) shape (6 trials) — hidden from the S2 tripwire because the dispatch silently skips an EST whose
  BOS sorts after it (§9.2 F17) → R8-2 opened; answered (b) → **S2b LANDED** (§9.10): the skip raises
  `FibEventOrderError`, byte-identical.

## 8. Handoff to session 2 — the implementation plan (from a fresh code audit + a cold review)

> Session 2 (2026-10-07): the plan is §9 — each point below is covered there (1 → §9.3 D1, 2 → D2, 3 → D4, 4 → D1 /
> D2 / D3, 5 → D1 + S4, 6 → D5 + S5, 7 → §9.2 F1, 8 → §9.4 / §9.5, 9 → §9.8, 10 → §9.9).

Session 2 writes the plan; it builds nothing (session 3 = the build). The RULE is §1.2 + §1.4 (decided — do not
re-open it; a question the code raises goes to the user as a new round, recorded in §5 / §6). The plan must cover:

1. **The limit, one routine for main and sub** — `P` by the starting trigger (§1.2), fixed once at the start (R2-4);
   `M` set in real time when a cycle's latest CTS anchor first passes `P` STRICTLY (Q2, N-b, N-f; the anchor fallback
   §1.1); within the limit nothing else changes (the dead-cycle walk, own-imbalance test, versions, `cross_failed`).
2. **The previous-sub rule (R2-1, R2-2, R3-5, R4-5, R4-6, R5-3)** — x−1 read by MOMENT, inclusive of x's start; an
   ended x−1 as of its `end_idx` (its frozen thresholds); the sub geometry runs on to the data edge, so the read goes
   through the end clip (the exported events are already clipped; the internal geometry is not).
3. **Unique subs numbered at their lifecycle START (R4-2, R4-4)** — a never-started sub gets no `sub_id`; its
   zero-length records keep `direction` + `starting_idx`, `sub_id` empty; an assert on the ordering; `trigger_sub_sid`
   untouched (audit: nothing reads it as a start order). `sub_id` is the join key on every sub event / zone / CSV row →
   a contract-level change: its own atomic commit (LANDMINES "Event Contract Rules" rule 3).
4. **The new fields** (§4 N-a): `cross_fib_limit_price` / `_cycle` / `_source` (KEPT, R4-3) on the structure and on
   every fib record's meta (CSV + hover); optional `cross_fib_limit_from_sub_id`. **`cts_frozen` on
   `CTS_ESTABLISHED`** (R4-1) — the frozen breakout threshold, read like `bos_frozen` (additive meta).
5. **Main joins the sub's logic** — cycle 0 single everywhere (Q6: a new H1 sid 0 cyc 0 fib + IP POI); the
   pre-established cross on main (I2); post-reversal handling = the sub's (I5); fill-as-of = the current candle (Q5);
   one meta format (Q8 — `scenario` / `cycle1_bos_anchor_idx` go: a rule-3 migration); `started_by` on main (N-c).
   Scenario 1 + the revert REMOVED (Q4), the retired rules moved verbatim into a "Retired logic (history)" appendix of
   `CROSS_CYCLE_FIB_SPEC.md` + GLOSSARY entries marked retired (R2-3, N-d).
6. **MS reads the same fib decision** (Q7; CROSS_CYCLE_FIB_SPEC §11b "Part B") — `P` inside every MS run (main: its
   own reversal's `bos_frozen`; a sub: handed in from the pool).
7. **The audit items found in session 1:** can an H1 reversal structure have a cycle live ACROSS its start (§3.4 round
   5 — the one cross-reversal ordering case; R4-2 orders it either way)? Every number in this doc re-checked against
   the code at plan time (memory `feedback_trace_every_bound_before_claiming.md`).
8. **Predictions before any build** — variant replays, one cause per `/compare` (§2 "the cascade": a different fib →
   zone → IP pick → MS's sd-proximity confirmation → structure → triggers → subs; Q6 and Q7 can move CTS
   confirmations). Baseline S_F1. Stage the build so each cause lands and is compared alone; the `sub_id` renumbering
   (item 3) is byte-identical on the window (no id changes) — land it first or last, alone.
9. **A cold review of the plan** (multi-agent is allowed for plan review — state the size + tokens first; ask if
   > ~2M or > 10 agents), then the user's approval.
10. **Knock-on:** PLAN_H's paused item-1 debugging points move (§2 last bullet) — re-derive them after the build.

## 9. The implementation plan (session 2, 2026-10-07)

**Status: cold-reviewed (§9.8); round 6 ANSWERED (§6 R6-1 … R6-7, every recommendation taken) + round 7 (R7-1 …
R7-4); APPROVED by the user 2026-10-07 (R7-4); session 3 = the build: **S1 LANDED** 2026-10-07, **S2 LANDED** 2026-10-08 (§9.10).** Written from a fresh code audit of HEAD `e63c3ea` + variant replays on S_F1. The rule is §1.2 + §1.4 (not
re-opened). The plan's AUTHOR'S RESOLUTIONS went to the user as round 6 and were all taken (§6). Times UTC (H1 i from the window's `_raw.csv`; M15 i = H1 i // 4 + 15 min × i % 4, the census
convention); M15 indices entity-absolute unless a sub's slice-local frame is named. "Window" = 2025-11-15 → 2026-01-20.

### 9.1 The audit — every site the change touches (HEAD line numbers are hints; anchor on the names)

**FibTracker / the shared routine (`zones/`)**

| Site | Today | Changes in |
|---|---|---|
| `fib_tracker.py` `FibTracker.__init__` 239 | `fib_mode` `"h1"` (default) / `"cross_cycle"`. h1-only state: `_cross_cycle_data`, `_scenario1`, `_prev_bos_outer`, `_bos_moment_by_cycle`. Shared by the sub path AND main's §11b path: `_dead_cycles`, `_bos_by_cycle`, `_cts_by_cycle` (written at CTS_CONFIRMED only, 1776), `_cross_version`; sub-only: `_m15_phase`, `_cts_confirmed_idx` | S2 (limits, `_latest_anchor`), S3 (limits required), S4 (one mode) |
| `_on_cts_established` 601–696; public `on_cts_established(event, df, bos_idx, bos_price, reversal_confirmed_idx, prev_bos_outer, prev_sd)` 583, `on_cts_updated(event, df, reversal_confirmed_idx)` 1267 | branches cross_cycle / sid 0 / sid ≥ 1; stashes `_prev_bos_outer` 672–675 | S4: one path; signatures become `on_cts_established(event, df, bos_idx, bos_price)` / `on_cts_updated(event, df)` |
| the h1-only handlers | `_handle_sid0_cts_established` 759 ("sid=0 cycle=0 NO FIB (simple flow)" 781–784); `_handle_sid1plus_cts_established` 803 (cycle ≥ 2 §11b peek 866–871); `_handle_cycle0_scenario1` 889; `_handle_cycle1_scenarios` 961 (revert 988–998, M-ceiling 1076–1084, meta `scenario` / `cycle1_bos_anchor_idx` 1107–1112); `_handle_sid0_cts_updated` 1409; `_handle_sid1plus_cts_updated` 1429; `_handle_cycle0_cts_updated` 1468; `_update_cycle1_main` 1633 (reads meta `cycle1_bos_anchor_idx` 1684; the 2026-09-28 assert 1717); `_create_updated_fib_state` 1731 and its only callee `_create_fib_retracement` 1136; `_handle_sid0_cts_confirmed` 1832; `_handle_sid1plus_cts_confirmed` 1858; `_handle_cycle0_cts_confirmed` 1912; `_should_revert_scenario1` 1950; `_deactivate_cycle0_fib` 1971; `_cross_allowed_for_target` 2055 (inclusive `>=` / `<=`, skips unconfirmed cycles); `_run_main_cross_check` 2086; `_maybe_activate_main_cross` 2110; `_c0_has_unfilled_now` 381 | S4: DELETED |
| `_activate_fib` 1162 | the h1-gated versioned-cross obsolete 1244–1253; the prev cycle's terminal stamped at THIS activation's moment (1221–1233) | S3b (R6-1b (ii): the prev cycle's terminal at its own cycle's establishment); S4 (gate deleted) |
| the sub path | `_handle_cross_cycle_cts_established` 698, `_handle_cross_cycle_cts_updated` 1332 (cycle-0 late activation 1343–1379; "unexpected phase" return 1383–1387), `_handle_cross_cycle_cts_confirmed` 1791 (sets `_m15_phase[(sid, n+1)] = "pre_established"` 1825), `_on_cts_threshold_updated` 2210, `_m15_cross_check` 2272 (own window filled → return 2342–2346), `_m15_create_cross` 2421, `_m15_extend_cross_anchor` 2489, `_m15_reactivate_cross_in_place` 2527, `_deactivate_cross` 2583, `_activate_or_update_single_m15` 2630 (main's §11b reaches it too) | S3 (the limit), S3b (R6-1b (i): the own-window-filled return 2342–2346 re-checks the single), S4 (THE path; `m15_` names go — R6-3 (vii)), S6 (the `fib_mode` stamps 735 / 1376 / 2464 / 2661) |
| `get_active_fib` 2670 | keys on `_current_cycle` (the last cycle that ACTIVATED a fib — lags when a cycle gets none) | S5: a public `representative_fib(sid, cycle)` |
| `_finalize_lifecycle_fields` 470 | per-cycle start / end / status; `_INVALIDATION_END_REASONS = {"scenario1_revert"}` 40 | S2 (limit meta), S4 (the set empties) |
| `select_fib_anchor_for_cycle` 102–217 | cycle-1 Scenario-2 wrapper; returns "intra" for sid 0 and every cycle ≠ 1 (176); callers `_handle_cycle1_scenarios` + MS's resolver | S4 (main caller gone), S5 (deleted) |
| `cross_cycle_fib.py` `resolve_cross_cycle_eligibility` 56 | the walk; `fill_as_of="snapshot"` + `prior_cached_liveness` + `snapshot_horizon_idx` serve the cycle-1 wrapper (main's FibTracker AND MS's resolver via `poi_zones.py:1373`); `target_ceiling` reserved, ignored | S5: snapshot policy + `target_ceiling` deleted |
| `structure_lifecycle.py` | `compute_reversal_idx_by_sid` | S2: + `compute_main_cross_fib_limits` |

**MarketStructure / structure engine (`structure/`)**

| Site | Today | Changes in |
|---|---|---|
| `market_structure.py` CTS_ESTABLISHED emit 1306–1318 | `st.cts_threshold` still holds cycle c−1's value (reset 1351); the winner's detection threshold `breakout_th` is a local of `_step_anchor` 842–857 | S2 (`cts_frozen` + its carrier) |
| reversal apply 1262–1272 | `STATE_CHANGED(to=reversal).meta["bos_frozen"]` = `st.reversal_bos_th_frozen` (None-able in code: `None if … is None`) | S2 (main's / a sub reversal's P, asserted not None) |
| the back-fill `_replay_step_no_patterns` 547–572 (`freeze_range` skips `_update_active_range` only) | still runs raw CTS updates and `_maybe_confirm_cts_via_proximity` 1786 on every back-fill candle; a proximity confirmation re-inits `cts_threshold = cts_price` 1994 and may emit a range-sync CTS_THRESHOLD_UPDATED 2001 | — (F14) |
| in-flight read `_refresh_poi_inners_for_cycle` 1820–1852 (at CTS_EST 1406, every CTS_UPDATED 1406 / 1542); `cycle0_data`, `_update_cycle0_data` 1855, `_lock_cycle0_data` 1909, `cts_established_moment_idx` 234 / 1645 | resolver `poi_zones.compute_poi_inners_for_cycle` 1298: anchors via `select_fib_anchor_for_cycle`, `evaluated_at=None`, no own-imbalance gate, ad-hoc `FibState(active=True)`, `select_ip_ic` errors re-raised as `IPSelectionError`; `df.attrs["timeframe"]` read inside `select_ip_ic` (`poi_zones.py:424`) | S5: deleted, replaced by the embedded tracker |
| proximity gate 1802–1813 | `cts_cycle_id > 0`: cycle 0 never confirms by proximity | — (F2) |
| `structure_engine.py` `_make_market_structure` 43 (resolvers by `setdefault` 64–67), `compute_structure` 105 (sid N = an independent MS run from `probe.starting_idx`), `compute_bounded_structure` 310 | — | S5 (tracker + P, REQUIRED keyword) |
| `unified_probe.py` Phase 2, MS built at 540 | its result reads two CTS_ESTABLISHED, cycle 0's CTS_CONFIRMED, a reversal before cycle 1 | S5 (passes "no P" explicitly) |
| `event_fields.processing_order_key` 130–135 | `(stamped_idx, type)`: CTS_ESTABLISHED / BOS_CONFIRMED / pattern-path CTS_UPDATED at their ANCHOR (LANDMINES "Event Sort Order Is a Dispatch Invariant"; `tests/test_event_order_pins.py` H1–H3) | — (F14) |

**The pool, the sweep, the sub build (`multitf/`)**

| Site | Today | Changes in |
|---|---|---|
| `sub_structure_pool.py` `get_or_create` 364 (`sub_id` = creation order), `TriggerRecord.sub_id` 165 ("never None"), `_by_id` 353, `get_by_id` 376, `all()` 380–382 (sorted by `sub_id`), `add_record` 399–402, `active_record` assert message 433, `identity` 217 (`trigger_sub_sid`) | — | S1 |
| `lifecycle_sweep.py` `_resolve_and_record` 270–386 | geometry (the sub's MS run) built at the FIRST FIRE, step 3 (306), BEFORE the record exists (336); `natural_reversal_idx` → spawn 315 + known ends 365–376; record start `max(finalize, trigger_idx, floor)` 350. Internal `sub_id` uses: spawn order key 315, absorb 325, `ended_by_sub_id` 360–362 / 446 / 454, `get_by_id` 390 / 463 / 488, `_touched`, the SUB_END order key + payload 459 / 484 → `_phase4_sub_end(sub_id)` 487 | S1 (`pool_idx`), S2 (P before the build) |
| the geometry callable chain | `lifecycle_sweep` `build_geometry(key, bos0_inner)` 143 / 306 / 584 → `orchestrator` closure 1013–1017 → `entity_df_mutation.build_or_get_geometry` 1166 (hit / miss at 1191; MS FIRST 1196, `pool.get_or_create` only on success 1207) → `_build_geometry` 1095 (`except (ValueError, IndexError)` 1154) → `structure_engine.compute_bounded_structure` 310 | S2 (P through the chain, stored after creation), S5 (P into the MS run) |
| `entity_df_mutation.py` `_FIB_META_IDX_KEYS` 165 | lists `cycle1_bos_anchor_idx` (the mirror's shift list) | S4 (rule-3 migration) |
| `entity_df_mutation.py` `_build_sibling_cts_ref_zone_from_pool` → `pool.get_by_id(r.sub_id)` 521 | runs DURING the sweep (phase 0, the counter / subsequent resolvers) | S1 |
| `entity_df_mutation.py` logs 949 / 1215; `render_sub_projection` 1222 (`started_by` = the first LIVE record by `(start_idx, seq)` 1256 / 1310) | — | S1 (labels), S2 (P to the projection) |
| `pooled_structure_build.project_to_window` 53 | `fib_mode="cross_cycle"` | S2 (P), S4 (`fib_mode` gone) |
| `sid_records.py` `build_sid_records_for_main` 19 | main SidRecords (attrs only, not exported) | S2 (R6-3 (iii): `started_by` on the main SidRecord's meta) |

**Orchestrator, exports, charts, guards**

| Site | Today | Changes in |
|---|---|---|
| `orchestrator.py` `_run_downstream_pipeline` 191 | `fib_mode` param 197 (H1 default `"h1"`), `FibTracker(fill_threshold=0.70)` 302, `_get_prev_bos_outer` 312–337, loop 339–375 (passes `reversal_idx` / `prev_bos_outer` / `prev_sd`) | S2 (`cross_fib_limits`), S4 (`fib_mode`, `_get_prev_bos_outer` gone) |
| `orchestrator.py` H1 call 543–547; `_run_multi_tf_dual` 844 (sweep 1005–1028; projections 1048–1065; the pool print 1034–1041 with `trigger_sub_sid`; `rendered.sort(key=(start_idx, sub_id))` 1050; never-started print 1063; WVMI join on `res.meta["sub_id"]` 807–816) | — | S1 (re-checked), S2 (limits; the sweep's P callable; the from-id remap) |
| `debug/export_sub_tables.py:34` | `_TRIGGER_COLUMNS = [f.name for f in fields(TriggerRecord) if f.name != "source_trigger"]` — EVERY field is a CSV column | S1 |
| `debug/export_fib_lifecycle.py` (meta verbatim; `version` / `fib_mode` via `.get`); `debug/export_events.py` + `structure_levels` (event meta verbatim) | — | S2, S4, S6 (the `fib_mode` column 26 / 60) |
| charts | `export_m15_chart.py:755` `int(tr.sub_id)` on EVERY record; `:464–468` `int(tr.ended_by_sub_id)` (the replacement break); `_recency_key` 353; `_zone_render.py:205`; the fib hover `_zone_render.fib_hover_status_text` 211–229 (status / life / IP — no meta) on both charts | S1; S2 (the hover line, R6-7) |
| guard tests (static — invisible to a "retired path raises" census) | `tests/test_event_meta_idx_keys.py`: `NON_INDEX_INT_META_KEYS` 367–372 + the int-value guard 654–700; fib_tracker `must_see` 322–324 (incl. `cycle1_bos_anchor_idx`); the inverse guard 390; `tests/test_event_fields.py:161`; `tests/test_meta_key_renames.py`; `tests/test_exception_scopes.py:22` (`compute_poi_inners_for_cycle`'s except scopes); `tests/test_cross_cycle_fib_routine.py` snapshot tests 61–133, 205–265 | S2, S4, S5 (§9.6) |

### 9.2 What the audit and the variants found

- **F1 — §8 item 7 answered: YES, by the code.** Each H1 sid is an independent MS run from its probe start
  (`compute_structure`), which lies before sid N−1's reversal apply; nothing ties sid N's cycle establishment to
  that apply. Window: sid 1's cycles 0 (EST 703, 12-29 05:00) and 1 (EST 748, 12-31 02:00) are established before
  its start; cycle 2 is established AT it (902, 01-09 12:00) only because the one `continuous` pattern 897 → 902
  broke both sid 0's `bos_threshold` 0.5736 and sid 1's `cts_threshold` 0.57376, which lies above it. Had sid 1's
  threshold been below 0.5736, or had (1,1) still been unconfirmed at 902, (1,1) would have stayed live across the
  start. A real M15 instance of the same mechanism: sub 2 (start 2470; cycle 0 EST 2368, next 2608). R4-2 orders it
  either way (§3.4 round 5).
- **F2 — PLAN_I §2 corrected:** Q6's new H1 (0,0) single (BOS 96, 11-20 22:00 → CTS 430, 12-10 20:00) gets NO POI
  on the window — 151 IC candidates, 11 ranked patterns, 3 touching the 61.8–80 zone, none both → no IP pick — and it
  can never move (0,0)'s confirmation (MS's proximity gate needs `cts_cycle_id > 0`, `market_structure.py:1808`). It
  IS drawn on the H1 chart (locked, not collapsed: one new fib trace, §9.5 c). No cascade (V4: POIs, events,
  triggers, subs keyed-equal).
- **F3 — PLAN_I §2 / §3.1 corrected:** the pre-established cross fires on main once on the window: (1,0) confirmed at
  727 (12-30 05:00), then the range-sync `CTS_THRESHOLD_UPDATED` at 741 (12-30 19:00, 0.57902 → 0.57899) creates
  (1,1)'s cross v0 from cycle 0, activated at 741 instead of 748, and Option A ends (1,0)'s new single at 741. All
  before sid 1's floor 902 → collapsed, not drawn; the fib CSV shows it.
- **F4 — three single behaviours exist today** (→ R6-1): (a) the sub's cycle-0 single — activates at CTS_ESTABLISHED
  if its own `[BOS, CTS]` holds an unfilled sd imbalance (formed by the moment), else at the first CTS_UPDATED that
  finds one; every update re-checks (deactivate / reactivate); (b) main's plain single (sid 0 cycle ≥ 1, sid ≥ 1 past
  the old ceiling) — the same without late activation; (c) the `cross_failed` single (`_activate_or_update_single_m15`;
  reached by the sub path AND by main's §11b path) — touched only while the own window is unfilled and the walk fails.
- **F5 — the `cross_failed` single goes stale** (→ R6-1b (i)): `_m15_cross_check` returns when the own window is filled
  (`fib_tracker.py:2342–2346`) without touching an existing single — it keeps its older CTS anchor and `active=True`
  while its own window holds no unfilled imbalance (a smaller window cannot be unfilled when the larger one is filled
  at the same horizon). S_F1: 0 occurrences (V2 log).
- **F6 — a sub's MS run happens at its first trigger's FIRE** (phase 0 builds the geometry before the record exists)
  → its P must be computable then (→ R6-2). The chain, link by link: the fire is at `trigger_idx`
  (`lifecycle_sweep.py:181–183`; a spawn at R, 315); the probe runs at step 2 (284–303); the geometry at step 3 (306);
  the record start = `max(finalize, trigger_idx, floor)` (350), floor = LOH(max(struct_start, cts_moment))
  (`parent_tables.py:146–149, 162–166`). At the fire the record start is known; so is every sub that starts at or
  before it — except in the cases R6-2 lists. Window: lags 2 (sub 0 463 → 1020, sub 4 3611 → 3621), both
  first_confluence (their P reads no previous sub); the creating record == the started_by record 8 / 8.
- **F7 — the probe's result does not depend on the POI read** (Phase 2 reads two CTS_ESTABLISHED, cycle 0's
  CTS_CONFIRMED — never by proximity — and a reversal before cycle 1). A probe MS gets "no P" (all single); a test
  pins the independence.
- **F8 — the latest-anchor fallback (N-f) never decides a cross on the window — a MEASURED invariant, not a code
  guarantee.** Chain: MS establishes cycle T only after T−1's CTS_CONFIRMED in EMISSION order (`establishing_new_cycle`,
  `market_structure.py:1298`); FibTracker decides in `processing_order_key` order, which places CTS_ESTABLISHED(T) at
  its ANCHOR; so the walk sees T−1's confirmed anchor only if CTS_CONFIRMED(T−1) sorts before EST(T)'s anchor — not
  guaranteed (F14). Window: holds for every cycle ≥ 1 EST (3 H1, 14 confluence, 5 counter; 0 / 34 ESTs have any
  fib-handled event of the same structure in [anchor, moment)). The fallback then only fills the exported
  `cross_fib_limit_cycle` of a structure whose limit-passing cycle never confirms (sub 7: M = 2 — cycle 2 passed P
  already at its establishment 4179, 01-19 10:45, and never confirmed; latest anchor 4222, 01-19 21:30).
- **F9 — MS's read differs from the drawn fib in THREE ways today:** (i) cross vs single — MS reads the single for
  every sub, H1 sid 0 and every main cycle ≠ 1, so main's §11b crosses are invisible to it too; (ii) the knowability
  cut (MS passes `evaluated_at=None` — the accepted M1 divergence, LANDMINES "Scenario 2 anchor agreement"); (iii) MS
  builds a fib FibTracker never creates or has deactivated (no own-imbalance gate, `active=True`). Q7 removes all
  three. V5: the inner differs at 20 of 371 refreshes (H1 (0,0) ×2 at 163 / 171, 11-25 17:00 / 11-26 01:00, and sub 3
  cycle 0 ×1 — cycle 0, never read; sub 5 cycle 1 ×4 at 4200–4222 — after sub 5's end 4159, never drawn; sub 7 cycle 1
  ×1 at 4157 (01-19 05:15) and cycle 2 ×12 at 4179–4222 (01-19 10:45 → 21:30) — the two crosses the limit keeps); 475
  proximity checks, 0 change their answer → 0 CSV cells, 0 chart traces.
- **F10 — main's P source moves** from the KL zone outer (`_get_prev_bos_outer`) to the applied reversal's
  `bos_frozen`. Equal on the window (0.5736). In general equal iff the previous sid's last BOS zone's base candle is
  its BOS anchor and every `BOS_THRESHOLD_UPDATED` of that cycle expanded THAT zone — KL expands only the side's
  ACTIVE zone of the same sid and never on a candle marked `reversal` (`kl_zones_v1.py:804–858`); not guaranteed by
  the code. The rule reads `bos_frozen`, so on another window S4 may change main's P where the two differ.
- **F11 — the PR1 contract (LANDMINES "The POI Resolver Fails Loudly") needs care in S5:** the new read reaches
  `select_ip_ic` only when the tracker holds an active fib, and a direct call raises `KeyError` on an unstamped frame
  (V5 suite: 3 `test_plan_h_landing_b.py` failures — the unstamped-frame and propagation tests).
- **F12 — `cts_frozen` equals the breakout's detection threshold on the window (19 / 19, a measurement):** at every
  cycle ≥ 1 CTS_ESTABLISHED of every MS run of the replay (probes included), `st.cts_threshold` before the reset ==
  the `breakout_th` the winning pattern was detected against at its anchor; cycle 0: None, 12 / 12. Window H1 values
  (0,1) 0.58448, (1,1) 0.57856, (1,2) 0.57376 = the last range-sync CTS_THRESHOLD_UPDATED before each EST (650,
  12-24 00:00; 746, 12-31 00:00; 880, 01-08 14:00). Not by construction: `freeze_range` only stops the range from
  expanding; a proximity confirmation inside the back-fill re-inits `cts_threshold` (F14); a breakout from no active
  range is detected against `breakout_th = None`.
- **F13 — x−1's thresholds read from its EVENTS == MS's own per-candle columns** (3 / 3 window reads): MS writes
  `cts_threshold` only at a confirmation (= the CTS price; `market_structure.py:1442 / 1994`) and by `range_sync`
  (always emitted, 1598), resets it at a new cycle (1351); `bos_threshold` only at BOS_CONFIRMED (1764) and by
  `BOS_THRESHOLD_UPDATED` (649 / 665 / 683 / 699).
- **F14 — the anchor-vs-moment order edge (all three review lenses; LATENT TODAY in the downstream).** The downstream
  tracker consumes `processing_order_key` order (CTS_ESTABLISHED at its anchor); MS emits by moment. They differ when
  a fib-handled event of the same structure has its stamped idx in an EST's [anchor, moment). Two mechanisms: (1)
  cycle T−1 confirms by proximity at a back-fill candle k of T's breakout window and T's anchor is ≤ k → the downstream
  handles EST(T) before CTS_CONFIRMED(T−1): the walk misses T−1 (`fib_tracker.py:2329–2333`, `cross_cycle_fib.py:175–
  176` — the N-f fallback would decide), then CONFIRMED(T−1) resets `_m15_phase[(sid, T)]` to `pre_established`
  (1825) and T's later CTS_UPDATEDs hit the "unexpected phase" return (1382–1387); (2) a CTS_THRESHOLD_UPDATED in
  [anchor, moment) (hazard H3) is a no-op downstream but would run a pre-established check in an at-emit tracker (S5),
  possibly creating a v0 + Option A end. Window: 0 / 34 (lenses A and B measured it too). Not proven unreachable →
  tripwires (D1, D5; R6-4) and a random-tail search at the build.
- **F15 — a late-activated single ends its PREVIOUS cycle late** (→ R6-1b (ii)): `_activate_fib` stamps the previous
  cycle's terminal at THIS activation's moment (`fib_tracker.py:1221–1233`, set-if-absent; the cycle pass-through end
  is fed only at finalize, 518–520). When cycle T's single activates late (on a CTS_UPDATED at u > T's EST), cycle
  T−1's fib ends at u although cycle T−1 ended at T's EST. Today reachable through the `cross_failed` single; S3 / S4
  make it reachable for every single past the limit. Window: 0.
- **F16 — MS confirms a CTS with an anchor AFTER its confirmation candle (the range back-fill look-ahead; found by the
  session-4 F14 search, 2026-10-08, §9.10; LATENT — 0 on the window) (→ R8-1).** `_step_anchor`'s no-winner branch
  confirms a range candidate at `i` (confirm `c`), back-fills `[i, min(c, D))` — the raw CTS update included
  (`freeze_range` freezes only the range) — and returns `i + 1`: the next anchors re-step those candles with the state
  already advanced to `c − 1`. A pullback pattern applying at `k` inside that window confirms the CTS MS holds, whose
  anchor `a` (raw-updated in the back-fill) can lie in `(k, c)`. MS emits `CTS_UPDATED`(raw, `a`) BEFORE
  `CTS_CONFIRMED`(idx `k`, anchor `a`); `processing_order_key` puts the confirmation first (`k < a`). Consequences:
  (i) TODAY's downstream locks the fib at the CTS known at `k` and drops the later update — the drawn fib's 100 % sits
  at an earlier candle than MS's confirmed CTS (the CTS_CONFIRMED price the KL CTS zone and the next cycle's cross
  walk `_cts_by_cycle` read); cycle ≥ 1 in both modes, cycle 0 in the sub mode (h1 mode draws no sid-0 cycle-0 fib);
  (ii) S5 (D5): the at-emit tracker sees the update first, and D5's end-of-run order tripwire (handled sequence ==
  `processing_order_key` order) RAISES. NOT shape (1): the S2 tripwire (`FibEventOrderError`) never fires on it.
  Real-candle example (NZD_USD M15, exact candles, a bearish run from 2025-03-24 10:30 UTC, BOS 0.5748): the range
  candidate 12:45 confirmed at 14:00; its back-fill raw-updated the CTS to 13:45 (low 0.57191); a later re-stepped
  anchor's pullback (`one_maru_continuous`) applied at 13:30 and confirmed the CTS with anchor 13:45; the cross_cycle
  fib locks at 13:00 (0.5723) in processing order, at 13:45 (0.57191) in emission order. Every search hit (6) is this
  mechanism: a pullback `CTS_CONFIRMED` at `k` inside a `RANGE_STARTED`'s `[start_idx, confirm_idx)` whose CTS
  anchor is the range's `cts_anchor_idx` > `k`. Window: 0 (the 30 exported CTS_CONFIRMED rows; V5's sequence check).
- **F17 — the downstream SILENTLY SKIPS a CTS_ESTABLISHED whose cycle's BOS anchor lies after its CTS anchor — and that
  hides the F14 (1) shape from the S2 tripwire (found by the session-4 60k WAVES run, 2026-10-08, §9.10; LATENT — 0 / 34
  on the window, 0 / 4,800 REAL slices) (→ R8-2).** The orchestrator loop hands a CTS_ESTABLISHED to the tracker only
  if its cycle's BOS_CONFIRMED was handled first (`if key in bos_anchor_by_cycle`, `orchestrator.py`), else drops it
  without a trace; `processing_order_key` puts BOS_CONFIRMED at its BOS anchor and CTS_ESTABLISHED at its CTS anchor,
  so a cycle whose BOS anchor is AFTER its CTS anchor loses its establishment — no fib record for that cycle at all
  (verified both modes, trial WAVES 413 / 1074), and the order tripwire behind it is never reached. MS's guard only
  asserts BOS anchor ≤ its moment (`_emit_bos_confirmed`). Two sources measured: (i) the F14 (1) shape itself — cycle
  T−1 confirmed by proximity at the back-fill candle `k` of cycle T's breakout, the breakout's pattern extreme = cycle
  T−1's own CTS candle `a < k` (no candle of the span goes beyond it: cycle T is established on the PREVIOUS cycle's
  CTS candle; no active range — the `cts_frozen` WARNING "detection threshold None" fired on each), and cycle T's BOS
  = the proximity candle `k`; 6 / 60,000 (e.g. WAVES 413 / 1074, M15, sd −1: cycle 1's CTS raw-updated to 46,
  cycle 2's breakout pattern 44 → 49, cycle 1 confirmed by proximity at 47, cycle 2 established at 49 with CTS
  anchor 46, BOS anchor 47 0.5785); (ii) a breakout span whose deepest pullback candle (the new BOS) comes after its
  extreme (e.g. WAVES 301 / 572, H1, sd −1: cycle 1 established at 45, CTS anchor 43 low 0.57645, BOS anchor 44 high
  0.60401 — a synthetic 250-pip wick); 8 ESTs (3 / 7,200 + 5 / 60,000). Every shape-(1) case the search produced was
  hidden this way (BOS(T) at `k` > `a`), so the S2 tripwire has not fired through the dispatch on any trial.

### 9.3 Design

**D1 — one FibTracker logic.** The sub path becomes THE path for every structure (an H1 sid, or a unique sub = its
sid 0); `fib_mode` and every h1-only handler / state / helper in §9.1 go (S4).
- Cycle 0: single (F4 (a)). Cycle T ≥ 1, the cross ALLOWED: the sub logic unchanged (the walk with fill to the current
  candle, the own-imbalance test, the versions, the pre-established phase on CTS_THRESHOLD_UPDATED with Option A,
  `cross_failed` → single). Cycle T ≥ 1, NOT allowed: single (R6-1 (d): the cycle-0 single's behaviour, labelled
  `via: "cross_fib_limit"`, R7-3), no pre-established phase. A single and a cross
  never coexist on one cycle (I8) — asserted at every cross creation and single activation (a test pins both orders).
- `_cross_allowed(sid, T)` = `limit.price is not None` and no cycle k < T whose LATEST CTS anchor (`_latest_anchor`,
  written at CTS_ESTABLISHED / CTS_UPDATED / CTS_CONFIRMED) is strictly beyond it — `beyond(p, P, sd)`: sd = +1
  `p > P`, sd = −1 `p < P`. Replaces `_cross_allowed_for_target`.
- `cross_fib_limits: Mapping[int, CrossFibLimit]`: optional in S2 (stamping only), REQUIRED from S3; a structure
  consulted without an entry raises `CrossFibLimitError` (a `RuntimeError` subclass — never swallowed by the
  geometry's `except (ValueError, IndexError)`). `CrossFibLimit` (frozen; `zones/cross_cycle_fib.py` with `beyond`):
  `price`, `source` (§4 N-a values + R6-3 (vi)'s), `from_pool_idx` (internal) / `from_sub_id` (the final id, set after
  S1's numbering — D2), `started_by` (main only; a sub's comes from the mirror).
- The order tripwire (F14 (1)): at CTS_ESTABLISHED of cycle T ≥ 1 the tracker requires (sid, T−1)'s CTS_CONFIRMED
  handled first, else it raises `FibEventOrderError` (a `RuntimeError`; R6-4). Lands in S2 (it never fires on the
  window → behaviour-neutral).
- Meta stamped on every fib record at `_finalize_lifecycle_fields`: `cross_fib_limit_price`, `_source`, `_cycle` (R6-6: the structure's FINAL M), `_from_sub_id`; main also `started_by` (also on the main SidRecord's meta, R6-3 (iii)). Where the fields live: P /
  source / from on `PooledStructure.cross_fib_limit` (a sub) and the tracker's `_limits` (main); M on the tracker
  (computed from `_latest_anchor`), exported only through the fib records (no structure-level CSV exists for main;
  the subs CSV is not changed — §4 amended).
- Meta format = the sub's (Q8): main's cross records carry `version` / `fib_mode` / `cross_start_cycle` and lose
  `scenario` / `cycle1_bos_anchor_idx` / `obsolete_reason` (the h1-gated obsolete goes); main singles lose `flow`
  (`"simple"`), `scenario1`, `scenario` — every key the sub path does not write (a consequence of "adopts the sub's
  meta", stated here because Q8 named only `scenario` / `cycle1_bos_anchor_idx`). Every record carries `fib_mode:
  "cross_cycle"` through S5 (R6-3 (iv)); S6 retires the key after the build (R7-2, §9.4). Rule-3 migration (LANDMINES "Event Contract Rules"), ONE commit: the emitter, the reader
  `_update_cycle1_main` (deleted), the shift-list entry `_FIB_META_IDX_KEYS` (`entity_df_mutation.py:165`), the guard
  tests (`test_event_meta_idx_keys` `must_see` 322–324 + the inverse guard 390, `test_event_fields.py:161`,
  `test_meta_key_renames`), the docs.
- `_INVALIDATION_END_REASONS` → `frozenset()`; the `disappeared` status and its chart / POI filters stay as a generic
  hook with no producer (R6-3 (v)).

**D2 — P by the starting trigger (S2)**, one helper per source, all read off events:
- main: `compute_main_cross_fib_limits(events)` (`structure_lifecycle.py`): per sid, `started_by` first — "reversal"
  iff the previous sid's `STATE_CHANGED(to=reversal)` exists, else "start_trading" (I1: the variation attaches to the
  trigger, not the sid number); "start_trading" → `CrossFibLimit(None, "none")`; "reversal" → that event's
  `bos_frozen` (asserted not None, ≤ 1 per sid, and == the OPEN WATCH's frozen barrier — the sid's last
  `REVERSAL_WATCH_START` at or before the reversal; §1.2 amended: the watch makes them equal), source
  `"reversal_bos_frozen"`. [S2 landing review, lens A M1: the plan first named "the applied `REVERSAL_CANDIDATE` —
  the one whose `apply_idx` is the realised reversal", but a later anchor's reversal WINNER can apply BEFORE the
  watch's pending candidate (MARKET_STRUCTURE_SPEC's later-winner case; `test_ms_reversal_on_expiry`'s
  `_later_winner_rows`: reversal at 8, the only candidate applies at 9) — no candidate sits at the realised apply and
  the assert would have stopped the replay. The watch is always emitted (`REVERSAL_WATCH_START`) and its frozen
  barrier IS the reversal's `bos_frozen`: equal on that fixture (sd ±1) and on the window (watch 897, 0.5736).]
- sub, `multitf/cross_fib_limit.py` (= `plan_i_inputs/cross_fib_limit_proto.py`, cleaned): the sweep's
  `_resolve_and_record` computes the record start `max(finalize, trigger_idx, floor)` ONCE, before step 3 (one formula
  for P and the record), and on a geometry MISS (`pool.get(key) is None`) calls an INJECTED
  `resolve_cross_fib_limit(trigger, record_start, reversing_sub) -> CrossFibLimit` before the build. The chain:
  `build_geometry(key, bos0_inner, cross_fib_limit)` → the orchestrator closure → `build_or_get_geometry(...,
  cross_fib_limit=)` → `_build_geometry(...)` → `compute_bounded_structure(..., cross_fib_limit=)` (S5 uses it inside
  MS); after `pool.get_or_create` → `sub.cross_fib_limit = P` (a new `PooledStructure` field, no usable default;
  `render_sub_projection` asserts it is set). The sweep test stubs of `build_geometry` change signature (§9.6).
- the sub rules (§1.2): reversal → the reversing sub's geometry `STATE_CHANGED(to=reversal)` `bos_frozen` (its idx +
  `slice_begin` == the reversing sub's `natural_reversal_idx`, asserted; not None); first_confluence on parent cycle 0
  → none; c ≥ 1 → the parent's CTS_ESTABLISHED (S, c) `meta["cts_frozen"]` (D3; asserted not None); first_counter /
  subsequent_* → x−1 at s = the record start: the sub with the largest (known start, `pool_idx`) among the subs whose
  known start ≤ s — known start = `start_idx`, else the min `start_idx` of its non-zero-length records (it can still
  move or vanish later in the cases R6-2 lists; the tripwire catches them); read x−1's geometry events by MOMENT
  (`knowable_at_idx` + `slice_begin`) up to `min(s, x−1.end_idx)` when the end is known — inclusive (R4-5), post-end
  events excluded (R4-6); at an equal moment the LATER-EMITTED event wins (list order); same direction → the most
  recent CONFIRMED cycle's `cts_threshold` (its CTS_CONFIRMED price, then that cycle's CTS_THRESHOLD_UPDATED), else its
  latest CTS anchor price (R2-b); opposite → the latest BOS_CONFIRMED / BOS_THRESHOLD_UPDATED price, else its starting
  candle's extreme (R2-2). No x−1 → none. A same-direction x−1 with no CTS anchor at all by s → no P, source
  `prev_sub_no_cts`, a WARNING (R6-3 (vi)).
- the final id (S1 ∘ S2): `CrossFibLimit.from_pool_idx` is recorded at the fire; after S1's numbering (inside
  `_after_sweep`), each `sub.cross_fib_limit` is replaced with `from_sub_id` = that sub's final id (None + a WARNING if
  it never started) BEFORE any `render_sub_projection`.
- **The R6-2 tripwire** (after the sweep and the numbering, in the orchestrator — it needs the H1 events and the
  pool): for every started sub whose started_by record is not its creating record, or whose type reads x−1, re-derive
  P from the started_by record at the sub's `start_idx` with the start-numbered x−1 (a reversal's reversing sub = the
  unique sub with `natural_reversal_idx == rec.trigger_idx`, direction == −x, a live record at R — asserted unique);
  compare (price, source, from_sub_id); a mismatch logs `WARNING [cross_fib_limit] …` (greppable by `/compare`) and
  the stored P stays (R6-2 (a)). Window: 0 mismatches expected (F6).
- the downstream: `render_sub_projection` → `project_to_window(cross_fib_limits={0: sub.cross_fib_limit})` →
  `_run_downstream_pipeline` → `FibTracker`; the H1 call passes `compute_main_cross_fib_limits(s_res.events)`.

**D3 — `cts_frozen` (S2):** CTS_ESTABLISHED meta `"cts_frozen": st.cts_threshold` captured before the reset — the key
ALWAYS present (None on cycle 0), read as `meta["cts_frozen"]`. A price → no `*_META_IDX_KEYS` entry. The detection
threshold is carried from `_step_anchor` to the apply in a state field set with the winner (`st.winner_break_th`,
cleared at the apply); at a cycle ≥ 1 EST, `cts_frozen != winner_break_th` (or `winner_break_th is None`) logs a
WARNING — not an assert (F12: equal 19 / 19 on the window, not by construction). Meaning, documented: the
`cts_threshold` MS holds when it establishes cycle T — the frozen breakout threshold except in the F14 (1) edge.

**D4 — `sub_id` at the lifecycle start (S1; R4-2, R4-4).** R3-2 proposed "number at the start (phase 2) OR renumber by
`(start_idx, creation order)` at the sweep's end"; the plan takes the second (one pass, the tie rule exactly R4-2's, no
id ever re-used mid-sweep) — R6-3 (viii).
- `pool_idx` (the creation index) is the FK during the sweep: a field on `PooledStructure` and `TriggerRecord`
  EXCLUDED from `_TRIGGER_COLUMNS` (like `source_trigger`; pinned) — S1 adds no CSV column. Every sweep-time use in
  §9.1 switches to it: spawn order key, absorb, SUB_END key + `_phase4_sub_end`, `get_by_id` → `get_by_pool_idx`,
  `add_record`, `_touched`, `ended_by_pool_idx` (internal), the sibling read `entity_df_mutation.py:521`, the sweep /
  geometry log lines (`pool#k`); `all()` sorts by `pool_idx`.
- Numbering runs inside `_Sweep._after_sweep()` BEFORE its asserts (every `SweepResult` consumer and test sees final
  ids): started subs sorted by `(start_idx, pool_idx)` → `sub_id` 0, 1, …; a never-started sub → `sub_id` None (its
  zero-length records carry None, keep `direction` + `starting_idx`); `ended_by_sub_id` = the ender's final id; the
  `sub_id` lookup map built (never-started subs absent). Asserts (replacing the tautological one): every live record's
  `sub_id` resolves to its sub; a never-started sub has only zero-length records; `from_sub_id` remapped (D2).
- Readers re-checked: `export_m15_chart.py:755` (skip a None `sub_id` — a zero-length record, drawn nowhere) and
  `:464–468` (`ended_by_sub_id` remapped); `export_sub_tables` (a None cell; pandas then renders the integer column as
  float, e.g. `3.0`, as `parent_bos_anchor_idx` already does — R6-3 (ix)); orchestrator 1034–1041 / 1050 / 1063 /
  807–816. `trigger_sub_sid` untouched — read by the pool print 1040, `TriggerRecord.identity` 217 (logs, asserts) and
  exported as a column; no reader treats it as a start order.
- Byte-identical on the window (ids unchanged: creation order == start order 8 / 8; 0 never-started subs).
  Contract-level → its own atomic commit.

**D5 — MS reads the same fib decision (S5; Q7).**
- `structure_engine._make_market_structure(…, cross_fib_limit)` — a REQUIRED keyword (no silent "no P") — builds
  `FibTracker(FibTrackerConfig(DEFAULT_FIB_LEVELS, FIB_FILL_THRESHOLD), cross_fib_limits={sid: P}, quiet=True)` and
  injects it (MS still imports nothing from `zones/`). One fill constant `FIB_FILL_THRESHOLD = 0.70` for MS's tracker
  and the downstream's (asserted equal); the embedded tracker logs nothing (its `[fib]` prints would flood run.log from
  every geometry and probe MS). P: main sid N ← sid N−1's run's `STATE_CHANGED(to=reversal)` `bos_frozen` inside
  `compute_structure`'s loop (sid 0: none); a sub's geometry ← D2's chain; the probe (`unified_probe.py:540`) passes
  `CrossFibLimit(None, "none")` explicitly (F7). `compute_bounded_structure(…, cross_fib_limit)` REQUIRED.
  `MarketStructure(fib_tracker=None)` (direct construction, 13 test files) = BOS inner only, exactly as
  `poi_inners_resolver=None` today.
- MS feeds the tracker at emit: CTS_ESTABLISHED (after the cycle's BOS_CONFIRMED, with `st.bos`), every CTS_UPDATED
  (raw + pattern path), CTS_CONFIRMED, CTS_THRESHOLD_UPDATED. Errors from those calls re-raise as a `RuntimeError`
  subclass (like `IPSelectionError`) so the geometry's `except (ValueError, IndexError)` cannot drop the sub (a test
  mirrors `test_an_ip_error_inside_a_sub_geometry_propagates`).
- The read at `_refresh_poi_inners_for_cycle`: `df.attrs["timeframe"]` read FIRST on every refresh (no-fib ones
  included), inside the `try` that re-raises as `IPSelectionError` (cause `KeyError`); then
  `tracker.representative_fib(sid, cycle)` (public; the active latest cross, else the active single, else None → BOS
  inner only); then the injected `poi_inner_for_fib(df, fib, fill_threshold)` (`poi_zones`: `select_ip_ic` → the pick's
  inner).
- THE ORDER (F14): the canonical order is the downstream's `processing_order_key` (the drawn fib is the reference); MS
  feeds by emission. The two are identical iff no fib-handled event of a structure has its stamped idx in one of its
  CTS_ESTABLISHED's [anchor, moment) — a MEASURED invariant (0 / 34 on the window), not a construction. Tripwire: at
  the end of every MS run, the tracker's handled sequence == the same events sorted by `processing_order_key` (else
  `FibEventOrderError` — R6-4); the S5 agreement test runs the V5 sequence check on fixtures. The build
  runs a random-tail search for the F14 shapes before S5 lands; a hit → a new round (which order wins).
- Deleted: `cycle0_data`, `_update_cycle0_data`, `_lock_cycle0_data`, `cts_established_moment_idx`,
  `select_fib_anchor_for_cycle`, `compute_poi_inners_for_cycle`, the snapshot fill-as-of (LANDMINES "Scenario 2 anchor
  agreement" — the M1 divergence — retires).

### 9.4 Stages — each ONE cause, each its own commit + `/compare` (keyed scripts; name the save)

| Stage | Cause | Code | Prediction (§9.5) |
|---|---|---|---|
| **S1** | `sub_id` numbered at the lifecycle start (R4-2, R4-4) — contract | D4 | **byte-identical**: 27 / 27 CSVs, the 3 figures equal (`cmp_fig_full.py`); only run.log's sweep lines change |
| **S2** | the new fields, NOT applied (`cts_frozen`; `cross_fib_limit_*` + main `started_by` on fib meta; the hover line, R6-7 — rendered for the user's review before S2 lands) + the behaviour-neutral tripwires (F14 order, D3 WARNING) | D2, D3, D1's stamping | §9.5 a |
| **S2b** (R8-2) | the dispatch's silent skip made loud (F17): a CTS_ESTABLISHED before its cycle's BOS_CONFIRMED raises `FibEventOrderError` | `orchestrator._run_downstream_pipeline`'s loop | **byte-identical** (0 / 34 ESTs on the window sort before their BOS): 27 / 27 CSVs == S2's, 3 / 3 figures |
| **S3** | the limit APPLIED to subs (drawn fib; MS unchanged) | D1's `_cross_allowed` + the past-limit single in the sub handlers; limits REQUIRED | §9.5 b |
| **S3b** (R6-1b) | the single's two latent quirks fixed (F5, F15) | `_m15_cross_check` 2342–2346; `_activate_fib`'s prev-cycle terminal | 0 cells on the window (F5 / F15: 0 occurrences) |
| **S4** | main joins the one path (R6-5: Q4, Q5, Q6, I2, I5, Q8, main's P = `bos_frozen`) | D1 (h1 deleted), the H1 call, the rule-3 migration | §9.5 c |
| **S5** | MS reads the same fib decision (Q7) | D5 | §9.5 d: 0 cells, 0 traces |
| **S6** (after the build, R7-2) | retire the `fib_mode` meta key (no other mode remains) — a rule-3 migration | the stamps (sub path), `export_fib_lifecycle.py`'s column, its readers / guards / docs | FORMAT-ONLY: every fib row of the 3 fib CSVs loses `fib_mode` (meta key + column; H1 5, confluence 21, counter 8 rows after S4) |

Docs move with their stage (code + spec + skill + memory in one commit):
- **S1:** PART4 §17 (identity, numbering); ARCHITECTURE 370 (FK `sub_id`); LANDMINES 819 (`pool.get_by_id(r.sub_id).geometry`); `.claude/skills/compare/SKILL.md:476` ("`sub_id`s are 0–7 in creation order"); GLOSSARY.
- **S2:** ARCHITECTURE (event meta `cts_frozen`); GLOSSARY (`cts_frozen`, `cross_fib_limit_*`, "beyond", "latest CTS anchor"); CHARTING_SPEC (the hover line).
- **S3:** CROSS_CYCLE_FIB_SPEC (the limit).
- **S3b:** CROSS_CYCLE_FIB_SPEC / FIB_LIFECYCLE_SPEC — the single re-checks its own window on every update; a
  late-activated single ends its previous cycle at its own cycle's establishment (whether today's text describes either
  quirk is checked at the build).
- **S4:** the CROSS_CYCLE_FIB_SPEC rewrite plus its "Retired logic (history)" appendix: Scenario 1 / 2 / 3, the revert, the ceiling `M` on `P_rev`, main's snapshot fill-as-of, verbatim, with the last commit that ran them = S4's parent (R2-3). Also FIB_LIFECYCLE_SPEC §5 / §6 / §10 / §15; LANDMINES 304 ("`_activate_fib`'s Versioned-Cross Obsolete Is H1-Only"), 677–700 (`_m15_phase` names), 1012 and ARCHITECTURE 513 (`fib_mode`), the sd-direction sites 1737 / 1765–1789; IMBALANCE_FILL_SEMANTICS' call-site matrix (FibTracker rows 224–243); GLOSSARY (retired entries, with pointers).
- **S5:** the appendix gains MS's Scenario-2 in-flight read and the M1 divergence, citing S5's parent. Also MARKET_STRUCTURE_SPEC ("Snapshot fib on SUBS", "Snapshot vs per-candle"); LANDMINES "Scenario 2 anchor agreement" (retired), "The POI Resolver Fails Loudly" (re-pointed), 1774, 2388–2398; ARCHITECTURE 192–197; IMBALANCE_FILL_SEMANTICS 78–95 / 265–279; POI_ZONES_SPEC.
- **S6:** the `fib_mode` meta key and fib-CSV column retired wherever S4's rewrite (CROSS_CYCLE_FIB_SPEC,
  FIB_LIFECYCLE_SPEC) or GLOSSARY lists them.

S4 as one stage is R6-5 (its causes, each with its own delta line, are listed there and in §9.5 c).

### 9.5 Predictions — every delta, by variant (CSVs: `plan_h_inputs/variant_report.py` + `plan_i_inputs/variant_report_i.py`; figures: `plan_e_inputs/review_scripts/cmp_fig_full.py` + a by-name trace multiset)

**a) S2 = V2 vs S_F1** — 7 CSVs, all FORMAT-ONLY (non-meta columns row-for-row equal; POIs / events / triggers / subs
keyed-equal); the 3 figures equal in V2 (V2 had no hover line — with R6-7 every drawn fib's hover gains one line,
counted at the build with `cmp_hover.py`):
- fib CSVs, 4 keys added on every row — H1 3 rows (+ `started_by`: `start_trading` ×1, `reversal` ×2), confluence 21,
  counter 8;
- events CSVs, `cts_frozen` added on every CTS_ESTABLISHED row — H1 5 (None ×2, 0.58448, 0.57856, 0.57376),
  confluence 21 (None ×7), counter 8 (None ×3); `structure_levels.csv` 5 rows (levels carry the event meta);
- the values (V2, per structure): H1 sid 0 `none`; sid 1 `reversal_bos_frozen` 0.5736, M = 2; sub 0 `none`; subs 1 /
  2 / 3 `reversal_bos_frozen` 0.57877 / 0.57926 / 0.58156 (from subs 0 / 1 / 2; M 2 / 0 / 1); sub 4
  `parent_cts_threshold` 0.57376 (M 0); subs 5 / 6 / 7 `prev_sub_bos_threshold` 0.58105 / 0.57112 / 0.57827 (from subs
  4 / 5 / 6; M None / None / 2) = §4's window values.

**b) S3 = V3 vs V2** — the fib / POI / IP-candidate CSVs of both lenses; MS untouched (events, triggers, subs equal):
- fib (confluence): the 6 cross records → 6 singles — sub 0 cycles 1–4 (no P), sub 2 cycle 1 (M = 0), sub 3 cycle 2
  (M = 1); counter: sub 3 cycle 2. Kept rows: the previous cycle's single now ends at the new single's activation —
  sub 0 c0 934 → 1020 (11-28 15:30 → 12-01 13:00), sub 2 c0 2600 → 2608 (12-24 00:00 → 12-24 02:00), sub 3 c1 2967 →
  2992 (12-30 19:45 → 12-31 02:00) (Option A goes with the cross); under R6-1 (d) every single the limit forces carries `via:
  "cross_fib_limit"` (R7-3) — exactly V3c's 9 confluence + 2 counter rows (V3c labelled them `cross_failed`): the 6 new
  singles and the 3 former `cross_failed` singles now past the limit (sub 1 c3, c4; sub 3 c3; the counter lens: sub 3
  c2, c3);
- POIs (IC): sub 0 c2 none → 1117 (12-02 13:15), c3 687 (11-26 01:45) → 1604 (12-09 15:00), c4 687 → 1772 (12-11
  09:00); sub 2 c1 2442 (12-22 08:30) → none; sub 3 c2 2837 (12-29 11:15) → 2969 (12-30 20:15), both lenses.
  IP-candidate rows: confluence 2985 → 1283, counter 409 → 332;
- figures: confluence 274 traces (8 "M15 POI zone" + 8 "M15 fib" replaced), shapes 258 → 264 (−42 / +48); counter 135
  traces (2 + 2 replaced), shapes 120 (−8 / +8); H1 equal.

**c) S4 = V4 vs V3** — the H1 fib CSV (3 → 5 rows), the H1 IP-candidate CSV (78 → 236), the H1 figure:
- behaviour, one line per cause: Q6 → + (0,0) single 96 → 430, start 115 (11-21 17:00), end 652 `new_cycle` (12-24
  02:00), ended, locked, no POI; Q4 → + (1,0) single 689 → 710, collapsed (end 741 `new_cycle`); I2 → (1,1)'s cross v0
  activated 741 (12-30 19:00; was 748, 12-31 02:00), (1,0) ended there (Option A); the h1-gated obsolete's deletion →
  (1,1) end_reason `new_cycle` → `next_cycle`, still collapsed; I5 → (1,2) a `cross_failed` single (the walk from
  target 2 stops at the dead cycle 1), same anchors 826 → 905 and lifecycle; Q5 and the P source → no delta (0.5736
  both ways);
- format (Q8): (1,1) `scenario` / `cycle1_bos_anchor_idx` / `obsolete_reason` removed, `version` 0 (CSV cell '' →
  `0.0`) / `fib_mode` / `cross_start_cycle` 0 added; (0,1) `flow` → `fib_mode` + `via: "cross_fib_limit"` (sid 0 has no `P`; by R7-3's rule — no variant labelled H1,
  V3c labelled the subs only); (1,2) + `fib_mode`,
  `via: cross_failed`;
- figure: H1 80 → 81 traces (+ `fib:sid0:c0`, its hover "ended (locked) / life=[115, 652] new_cycle / IP: none (no_pattern)" + S2's limit
  line, R6-7 — no `P`: "limit: none (…)"),
  shapes 230 → 235 (+5, its levels); M15 figures equal;
- POIs, events, triggers, subs, WVMI, zone proximity: equal (no new POI → no cascade).

**d) S5 = V5 vs V4** — every CSV md5-equal, every figure equal (full JSON). Evidence it is not dead code (F9): the
MS-embedded tracker's decision == the downstream's at every handled event (H1 sid 0 52 / 52, sid 1 24 / 24; each of the
8 subs' downstream sequence is an exact prefix of its MS sequence — the geometry runs past the sub's end); the inner
differs at 20 / 371 refreshes; 0 / 475 proximity checks change their answer.

**Not predicted = stop:** any delta outside these lists stops the stage (name the mechanism, ask).

### 9.6 Test census

Full suite inside trees where the retired path RAISES (`variant_patch_i.py` V3x / V4x / V5x; set differences between
them), plus a grep pass for the static guards that method cannot see:

| Stage | Moved tests (raise census) | + static / signature (grep) |
|---|---|---|
| S1 | not run (needs the implementation); grep of creation-time id pins (MATCHING LINES): test_sub_structure_pool 29, test_lifecycle_sweep_predicted_table 9, test_lifecycle_sweep_unit 6, test_geometry_builder 6, 4 more files × 1 | the `_TRIGGER_COLUMNS` exclusion pin (new) |
| S2 | — | `test_event_meta_idx_keys` `NON_INDEX_INT_META_KEYS` (+ `cross_fib_limit_cycle`, `cross_fib_limit_from_sub_id`) and its int-value guard 654–700; the sweep's `build_geometry` stubs (new signature) |
| S3 (a limit REQUIRED) | 26 — test_e4_simulation 11, test_cross_cycle_fib 7, test_fib_never_established_cap 4, test_imbalance_c3_knowability 2, test_plan_e_role_pins 1, test_e3a_mutation_pins 1 | — |
| S4 (h1 retired) | 111 — test_e3a_mutation_pins 25, test_poi_activation_moment 13, test_imbalance_c3_knowability 11, test_plan_h_landing_b 11, test_plan_g_wvmi 11, test_e4_simulation 11, test_main_versioned_cross 10, test_orchestrator_reversal_source 4, test_wvmi_lp_bound 4, test_zone_proximity_reversal_cap 3, test_plan_e_role_pins 3, test_institutional_pattern 2, test_meta_key_renames 1, test_cross_cycle_fib 1, test_fib_chart_render 1 | `test_event_meta_idx_keys` `must_see` 322–324 + inverse guard 390; `test_event_fields.py:161`; `test_meta_key_renames`; the `_m15_*` renames (test_cross_cycle_fib 6, test_imbalance_c3_knowability 1, test_cross_cycle_fib_routine 1); the handler signatures; stale `_NOT_MIRRORED` comments |
| S5 (old resolver retired) | 22 — test_plan_h_landing_b 8 (incl. the 3 PR1 tests, F11), test_imbalance_c3_knowability 4, test_e3a_mutation_pins 4, test_ms_cts_update_no_regress 3, test_plan_e_role_pins 3 | `test_cross_cycle_fib_routine` snapshot tests 61–133 / 205–265; `test_exception_scopes.py:22` |
| S3b | — (0 on the window) | the two constructed pins (below) |
| S6 | — (grep: no test reads the `fib_mode` meta key or CSV column; the `fib_mode=` parameter's tests move in S4) | `export_fib_lifecycle.py`'s column list |

The lists: `plan_i_inputs/census/census_S3.txt` / `_S4` / `_S5` (26 / 111 / 22 test ids). Each moved test is classified at the build as re-pin /
re-derive / retire (memory `feedback_cold_review_plans`: re-derive a pin in a tree with the old rule when the change
strips its mechanism). Tests the build adds: the limit (strict beyond on both sides sd = ±1; M = 0 / 1 / 2 / not
reached; the latest-anchor fallback); each P source incl. the x−1 fallbacks, an ENDED x−1 read up to its end, the
INCLUSIVE start candle (R4-5 — an x−1 threshold event ON x's start candle), a pattern / watch pending at x−1's end
(R5-3); the R6-2 tripwire (a constructed lag case); the past-limit single (late activation on an update); S3b: the stale `cross_failed` single re-checked when its own
window fills at a CTS_UPDATED (F5), and a late-activated single ending its previous cycle at its own cycle's
CTS_ESTABLISHED, not at the activation (F15); the
pre-established cross refused past the limit; no single + cross on one cycle (I8); main's (0,0) single; main's
pre-established cross clamped by the floor (R2-5); `cts_frozen` + the D3 WARNING; `sub_id` start order + a
never-started sub (None) + `from_sub_id` remapped; the order tripwire on a fixture with the F14 shape (built from the
mechanism, not found by sampling); MS ↔ downstream agreement (the V5 sequence check on a fixture with a cross); the
probe's independence from the read (F7); PR1 through the new read on a frame with no active fib; a tracker error inside
a sub geometry propagates. Mutation lens at each landing (value / swap / wrong-accessor mutants; every new
`if sd == 1` gets an sd = −1 pin; ≥ 3 elements in order pins).

### 9.7 Tools (`plan_i_inputs/`)

`variant_patch_i.py TREE V2|V3|V3c|V4|V5|V3x|V4x|V5x` (cumulative minimal patches on a `git archive` of `e63c3ea`;
exact single-match edits), `cross_fib_limit_proto.py` (the P prototype the patch copies in), `replay_variant_i.py TREE
LOG.json` (cwd = `%TEMP%/pi/<v>` with `oanda.cfg`; dumps the variant log), `variant_report_i.py BASE CUR` (meta-aware
CSV diff) + `plan_h_inputs/variant_report.py` (keyed POIs / events / triggers / subs) +
`plan_e_inputs/review_scripts/cmp_fig_full.py` (positional: BASE_DIR CUR_DEBUG CUR_CHARTS). V0 (unpatched) reproduced
S_F1 27 / 27 CSVs and 3 / 3 figures; replay walls 78–98 s (four in parallel).

### 9.8 Cold review (2026-10-07; 3 read-only lenses in parallel, fresh agents; 930,594 subagent tokens)

| Lens | Tokens | BLOCKER / MAJOR / MINOR | The findings that changed the plan |
|---|---|---|---|
| A — code-reference audit | 376,420 | 0 / 5 / 10 | `cycle1_bos_anchor_idx` also in `_FIB_META_IDX_KEYS` + 3 guard tests (D1 rule-3 list, §9.6); a `pool_idx` field would be EXPORTED (`_TRIGGER_COLUMNS`) → S1 not byte-identical as written (D4); 10 missed sweep-time `sub_id` sites incl. the sibling read 521 and `all()`'s sort (§9.1, D4); "same order by construction" false (F8, F14, D5); F12's mechanism overstated (F12, D3); the geometry chain (D2); the probe as an S5 site; census gaps of the raise method (§9.6); per-stage doc sites (§9.4) |
| B — implement on paper | 379,074 | 0 / 13 / 13 | the same three S1 / F14 / chain items; `from_sub_id`'s final id (D2); tracker errors swallowed by the geometry's except (D5); defaults of the new MS arguments (D5); PR1 on an unstamped frame with no fib (D5); the missing CHART predictions (§9.5 b / c — measured); the (1,1) format items (§9.5 c); the late-activation end (F15); R6-1b staging (S3b); required-from-S3 (D1); the tripwire's inputs (D2); `representative_fib`; one fill constant; the `cts_frozen` carrier; post-S4 signatures |
| C — decision coverage | 175,100 | 0 / 4 / 15 | the hover dropped from N-a / R3-4 without asking (→ R6-7); S4 carries seven causes (→ R6-5 widened); `cross_fib_limit_cycle` real-time in N-a vs FINAL here (→ R6-6); R6-1 (b) already ruled out by I-lean / I5; R6-4 implied by Q7 (now a statement + the order question); P from `STATE_CHANGED` vs §1.2's `REVERSAL_CANDIDATE` (D2, §1.2 amended); I1 → branch on `started_by` (D2); §1.4's pre-start bullet generalised to main; a source value for (vi); where M lives (D1, §4 amended); S5's retirements into the appendix (§9.4); UTC times + round-6 examples |

All findings folded (the round-6 items are the user's). **Pre-commit verification (after rounds 6–7; a 2-agent
workflow, 389,304 tokens):** lens B re-derived EVERY number of §9.2 / §9.5 from the S_F1 save + the variant outputs —
0 wrong (all 32 UTC times included; the one prediction no variant measures: H1 (0,1)'s `via` label, by R7-3's rule);
lens A's 11 MINOR consistency gaps (S3b / S6 missing from §9.1 / §9.4 docs / §9.6, two stale header phrases, §4's
"Optional", §9.5 c's hover, §9.9) folded. Verified OK by A (no change): every §9.1 FibTracker / MS /
orchestrator / pool / export / chart row within ±10 lines; F1, F2, F3, F4 (a)(b), F5, F6 (traced link by link), F7,
F10, F11, F12's values, F13; §9.5 a's counts.

### 9.9 Knock-on — PLAN_H's paused item-1 debugging points (§2's last bullet; re-derive after the build)

Measured (V3): of PLAN_H §8.5b C2's no-POI cycles (sub 0 cycles 0–2, all crosses from 454 then), cycle 2 — now the
single 1107 → 1245 — gains a POI (IC 1117, 12-02 13:15), cycle 1 — now the single 917 → 1020 — still has none, cycle
0 is unchanged (collapsed); sub 0 cycles 3 and 4 move their POI (687 → 1604, 687 → 1772); sub 3 cycle 2 (the T1
example) moves 2837 → 2969; sub 2 cycle 1 loses its POI (2442 → none). Sub 3 cycle 3 (no POI record) is unchanged by
the revamp (a single before and after). PLAN_H §8.9 resumes on the S5 save (S6, format-only — the `fib_mode` column — moves no debugging point).

### 9.10 Build record (session 3, 2026-10-07 → 08)

**S1 — `sub_id` numbered at the lifecycle start (D4) — LANDED** (its own commit; local, not pushed).
- Code: `sub_structure_pool` — `pool_idx` (the creation index) on `PooledStructure` and `TriggerRecord` + the
  record's `ended_by_pool_idx`; `get_by_pool_idx`; `number_subs_by_start()` (started subs by `(start_idx,
  pool_idx)` → 0, 1, …; never started → None, its records too; `ended_by_sub_id` from `ended_by_pool_idx`; runs once;
  a NEW `get_or_create` after it asserts); `all()` in `pool_idx` order. `lifecycle_sweep` — every sweep-time order
  key / FK / replacement link / log line on `pool_idx` (`pool#k`; every key's VALUE equals HEAD's old `sub_id` —
  lens A); `_after_sweep` numbers FIRST (`[sweep] number pool#k -> sub_id=j start_idx=…`), WARNs on a replacer that
  never started, asserts every live record resolves to its sub. `entity_df_mutation` — the sibling read
  `get_by_pool_idx(r.pool_idx)`; the reversal resolver's label and the geometry log `pool#k`;
  `render_sub_projection` asserts a `sub_id`. `export_sub_tables._TRIGGER_NOT_EXPORTED` (`source_trigger`,
  `pool_idx`, `ended_by_pool_idx`); `export_m15_chart` skips a record whose `sub_id` is None; the orchestrator's
  never-started print and `pool#k` on the `[pool] sub_id=` line.
- `/compare` vs S_F1 (`cmp_fig_full.py`): **byte-identical** — 27 / 27 CSVs md5-equal, 3 / 3 figures (full JSON)
  equal = §9.5's S1 prediction (twice: before and after the review fixes; replays 88.1 s / 46.6 s wall, fetch gate
  PASS, the log grep = the 4 unresolved + 2 degenerate-cycle lines as before). run.log: the sweep / geometry /
  probe-cache lines name `pool#k`, + 8 `[sweep] number pool#k -> sub_id=k` lines (creation == start order, 8 / 8),
  the `[pool] sub_id=` lines gain `pool#k`.
- Tests 1288 → 1298 (+10): the numbering (start ≠ creation order, a tie, a never-started sub, the `ended_by` remap,
  run-once / create-after guards — pool level; through the sweep with a replacement whose ender's `pool_idx` ≠ its
  id), the triggers-CSV header pinned literally + the float `sub_id` cell of a never-started sub (R7-2), the M15
  chart skipping a None-`sub_id` record, and the review's four (below). ~80 fixture updates (constructed records /
  subs take `pool_idx`; `_set_lifecycle` numbers as the sweep would) and re-pins: `test_sub_id_is_monotonic_and_stable`
  → `test_pool_idx_is_monotonic_and_stable` (PART4 §17.11 annotated); `get_by_id` → `get_by_pool_idx` + `get_by_id`
  after numbering; the field-order pin gains the two FKs; the geometry builder's "no index consumed" pins → `pool_idx`;
  the predicted table's id test → start order; the stubs / log greps that run mid-sweep → `pool_idx` / `pool#`; the
  two-chain invariants test → one pool per sweep (numbering runs once per pool).
- Landing review, 2 read-only lenses, **439,114 tokens**: **A** code audit (231,808) — 0 BLOCKER / 0 MAJOR / 4 MINOR /
  4 NIT, all folded: the never-started-replacer WARNING IS reachable — a same-candle chain where the incumbent's
  first equal-priority end entry names a co-starter that was itself replaced (latent since Plan C; HEAD exported that
  sub's creation id; register D.2, DECIDE; pinned); 3 stale lines in memory `project_sub_structure_pool_architecture`;
  the predicted table's id test; the invariant helper's `ended_by` remap; nits (an assert message, the synthetic
  fixtures' docstrings, `_recency_key` / CHARTING_SPEC wording, `pool#` on the pool print). Verified: no sweep-time
  `sub_id` read left; every post-sweep reader tolerates None; on a window where creation ≠ start order the deltas
  are ids / labels + the float column, as D4 intends. **B** mutation (207,306) — 41 mutants on the full suite: 31
  killed, 5 equivalent (N04, N11, S02, S03, S15 — `_touched` is dead state, pre-S1), 5 gaps → 4 tests (S13 / S14
  MAJOR: phase 1's "frozen while pending" `ended_by` copy was reached only with a `parent_end`; S18 the WARNING — and
  the collision test's `in {A.sub_id, B.sub_id}` had weakened to accept None; E02 the sibling read by `pool_idx`; E03
  the render assert) — re-run: 5 / 5 KILLED.

**S2 — the fields stamped, not applied (D2, D3, D1's stamping + the order tripwire) — LANDED** (its own commit; local,
not pushed). The hover (R6-7) rendered and APPROVED by the user 2026-10-08: "(a) approve as is, commit S2" (§6 R6-7;
spec copy part 9). One clarifying question first: the options named "the IP line" — PLAN_H's `IP <rank> · IC <idx>` /
`IP: none (…)` / `IP: not evaluated (…)` line, the status cell's line directly above the new limit line (the full
hover shown, e.g. H1 sid 0 cycle 1: `… life=[652, 902] reversal` / `IP 3A.iii · IC 607` / `limit: none
(start_trading)`). Re-checked before the commit: tests 1359 pass, fetch gate PASS on the replay of the final code.
- Code: `zones/cross_cycle_fib.py` `CrossFibLimit` (frozen; `price` / `source` ∈ `CROSS_FIB_LIMIT_SOURCES` /
  `from_pool_idx` / `from_sub_id` / `started_by`; price None ⟺ source `none` / `prev_sub_no_cts`) + `beyond`;
  `zones/structure_lifecycle.py` `compute_main_started_by` + `compute_main_cross_fib_limits` (the open-watch
  cross-check — D2 amended, lens A M1); MS `cts_frozen` on every CTS_ESTABLISHED (after `confirmed_at`, always present)
  + the carrier `MarketStructureState.winner_break_th` (set with the `_step_anchor` winner, cleared at the
  establishment) + `WARNING [ms] cts_frozen …`; NEW `multitf/cross_fib_limit.py` (`known_start`, `prev_sub_at_fire`,
  `read_prev_sub`, `reversal_limit`, `parent_cts_limit`, `cross_fib_limit_for` = the one formula, `check_cross_fib_limits`
  = the R6-2 tripwire — warn-only: an ambiguous reversing sub WARNs, never asserts); the sweep computes the record start
  ONCE and, on a geometry MISS, P via the injected `resolve_cross_fib_limit(trigger, record_start, reversing_sub)`
  (REQUIRED) handed to `build_geometry(key, bos0_inner, cross_fib_limit)` → `build_or_get_geometry(cross_fib_limit=)`
  (REQUIRED; a limit on a miss, None on a hit — asserted) → `PooledStructure.cross_fib_limit`; `_after_sweep` remaps
  `from_sub_id` (a never-started supplier → None + WARNING); `render_sub_projection` → `project_to_window(
  cross_fib_limits={0: P})` → `_run_downstream_pipeline(cross_fib_limits=)` → `FibTracker(cross_fib_limits=)`; the H1
  call passes `compute_main_cross_fib_limits(s_res.events)`; FibTracker `_latest_anchor` (EST / every UPD / CONF, both
  modes), `cross_fib_limit_cycle` (M, strict, FINAL), the stamp at `_finalize_lifecycle_fields` (`{**meta,
  price, source, cycle, from_sub_id[, started_by]}` — V2's key order), `FibEventOrderError` at a cycle ≥ 1
  CTS_ESTABLISHED without its cycle T−1's CTS_CONFIRMED (both modes); main SidRecord `meta["started_by"]`; the hover's
  limit line `_zone_render.fib_hover_limit_text`, the status cell's last line. S2 does NOT construct the MS-side
  chain (`_build_geometry` → `compute_bounded_structure`): S5.
- `/compare` vs S_F1: **every one of the 27 CSVs md5-equal to the session-2 variant V2** (`%TEMP%/pi/V2`); vs S_F1 exactly
  the 7 predicted, format-only (`variant_report_i.py`): §9.5 a's counts and per-structure values, all equal (H1 sid 0
  none; sid 1 0.5736 reversal_bos_frozen M 2; sub 0 none; subs 1 / 2 / 3 reversal_bos_frozen 0.57877 / 0.57926 /
  0.58156 from 0 / 1 / 2, M 2 / 0 / 1; sub 4 parent_cts_threshold 0.57376 M 0; subs 5 / 6 / 7 prev_sub_bos_threshold
  0.58105 / 0.57112 / 0.57827 from 4 / 5 / 6, M None / None / 2). Figures: trace counts + layouts equal; only the
  customdata of the 24 drawn fib traces (H1 2, confluence 16, counter 6) — each exactly one appended limit line
  (`cmp_hover.py` + a per-cell check). run.log: 8 `[sweep] cross_fib_limit pool#k …` + `[cross_fib_limit] tripwire:
  3 subs re-derived at their start, 0 mismatches`; no `WARNING [ms] cts_frozen`, no `FibEventOrderError`.
- Tests 1298 → 1359 (+61 in `test_plan_i_s2_cross_fib_limit.py`). Moved: the sweep / builder stubs and fixtures (the
  two new REQUIRED arguments — §9.6 named the sweep stubs; the builder fixtures were not censused); `NON_INDEX_INT_META_KEYS`
  + `cross_fib_limit_cycle` / `_from_sub_id`; and 7 h1-mode unit tests (test_e3a_mutation_pins ×3,
  test_imbalance_c3_knowability ×2, test_plan_e_role_pins ×2) that fed cycle 1's CTS_ESTABLISHED with no cycle-0
  CTS_CONFIRMED — an order MS never emits, caught by the new order tripwire, NOT in the census: each now feeds that
  CTS_CONFIRMED first, every original assertion unchanged (lens A: sound — the h1 handlers treat it as a no-op for
  those paths).
- Landing review, 2 read-only lenses, **613,095 tokens**: **A** code audit (335,832) — 0 BLOCKER / 1 MAJOR / 4 MINOR /
  NITs. **M1**: main's cross-check against "the applied `REVERSAL_CANDIDATE` at the realised apply" (D2's literal text)
  CRASHES on a legitimate stream — a later anchor's reversal winner applies before the watch's pending candidate
  (`test_ms_reversal_on_expiry._later_winner_rows`: reversal 8, the only candidate applies at 9) → the check now reads
  the OPEN WATCH (`REVERSAL_WATCH_START`), D2 / §1.2 amended, pinned sd ±1. m1 missing tests (R6-2 lag, the reversal
  lookup, R5-3) → added; m2 the hover's "why" reads the first LIVE record while P comes from the creating trigger →
  documented as a known limit (CHARTING_SPEC; the R6-2 tripwire WARNs wherever the two rules differ; 0 on the window);
  m3 an assert inside the warn-only tripwire → a WARNING; m4 docs (WORKFLOWS, POI_ZONES_SPEC, a FibTracker docstring,
  this record); nits folded (incl. the S1 GLOSSARY rows that ride in this commit). Verified: every P source = §1.2;
  behaviour-neutral (`_latest_anchor` / `_limits` / `cts_frozen` / the carrier are read only by the stamp / the
  WARNING); the order tripwire keys on `_cts_by_cycle` (written only at CTS_CONFIRMED), ties sort CONFIRMED first,
  the cap clip keeps CONFIRMED(T−1) whenever EST(T) is kept. **B** mutation (277,263) — 73 mutants on the full suite:
  49 killed, 4 equivalent (S04 / S05 / S07 the carrier, C01 `ev.idx` = the moment), 20 gaps (MAJOR: the end-to-end
  wiring E01 / O01; the tripwire's distinctions T01 / T04 / T06 / T07; the handlers' sd capture F05 / F06) → 15 tests
  (+ the watch versions of M05–M07); re-run with 6 new mutants on the watch check / the ambiguity count: 23 / 23
  KILLED. Fixture realism notes folded as comments (synthetic shapes named).
- Open (beyond S2): the order tripwire now RAISES in the downstream on any window with the F14 shape — the planned
  random-tail search for it (D5: "before S5") matters from S2 on; proposed for the start of the S3 session.

**Session 4 (2026-10-08) — the F14 random-tail search** (`plan_e_inputs/review_scripts/random_tail_search.py`,
extended: an `f14` record on every trial of every stream — shape (1) = the S2 tripwire replayed on
`processing_order_key` order, shape (2) = fib-handled events whose emission order ≠ that order, the shape-(1)
precondition = a proximity CTS_CONFIRMED inside the next cycle's breakout back-fill; two new streams: WAVES (synthetic
impulse / pullback / chop legs, pullbacks often ending in a zone-touching wick; H1 / M15) and REAL (random 150–900
candle slices of a year of NZD_USD H1 / M15 — `%TEMP%/pi/f14/NZD_USD_<tf>_2025.csv`, fetched 2025-01-01 →
2026-01-20 — from the slice's sd-side extreme in its first 1–29 candles, half with a ≤ 1-pip jitter)). Results: 0
errors; **shape (1) 0** (WAVES 7,200: cycles ≥ 2 in 753 trials, 1,198 proximity confirmations, the precondition 9 ×
— each EST's anchor its apply candle, after the confirmation; REAL 4,800: cycles ≥ 2 in 614, 443 proximity
confirmations, the precondition 0); **shape (2) 5** (WAVES 2, REAL 3) + 1 in the 690-trial yield runs = **F16**
(§9.2; one mechanism, all 6). → round 8 (R8-1), OPEN; S3 not started. R8-1 answered (a) — F16 parked to S5.
A further 60,000 WAVES trials (seeds 401–424 × 2,500): 0 errors; cycles ≥ 2 in 5,940; 9,598 proximity confirmations;
the shape-(1) precondition 121 × in 118 trials; shape (2) 42 trials (36 the F16 kind — 38 pairs) — and the other 6 the F14 (1)
shape itself (CTS_CONFIRMED(T−1) emitted before an EST(T) whose anchor precedes it), which the tripwire replay did NOT
flag: the dispatch had skipped those ESTs (their BOS sorts after them) → **F17** (§9.2). With a `skip` counter (an
EST whose cycle's BOS has not been handled — the orchestrator drops it) all three streams re-run (deterministic, the
same counts): REAL 0 / 4,800; WAVES 3 ESTs / 7,200; WAVES-B 11 ESTs in 10 trials / 60,000 (6 the hidden shape 1). →
R8-2, OPEN → answered (b) (§6).
Tools: `plan_i_inputs/f14_order_effect.py` (a trial's fib records in processing vs emission order, both modes) and
`plan_i_inputs/f14_slice.py` (MS on an exact real slice, the F16 example's times) — inputs to the S5 round.

**S2b — the dispatch's silent skip made loud (F17, R8-2) — LANDED** (its own commit; local, not pushed).
- Code: `orchestrator._run_downstream_pipeline` — a CTS_ESTABLISHED whose cycle's BOS_CONFIRMED has not been handled
  raises `FibEventOrderError` ("… handled before its cycle's BOS_CONFIRMED") instead of being dropped; the error's
  docstring, GLOSSARY, LANDMINES "Event Sort Order Is a Dispatch Invariant" (+ the parked MS side).
- `/compare` vs S2's outputs (`%TEMP%/pi/S2`, `cmp_fig_full.py`): **byte-identical** — 27 / 27 CSVs md5, 3 / 3 figures
  (full JSON) = the prediction; fetch gate PASS; replay 44.5 s wall; the log grep = the 4 unresolved + 2
  degenerate-cycle lines as before.
- Tests 1359 → 1364 (+5, `test_plan_i_s2b_dispatch_order.py`): the raise on a constructed BOS-after-CTS stream (and its
  processing order), the BOS-first tie / before cases pass, the `RuntimeError` class, and the F14 (1) stream of a REAL
  MS run (WAVES 413 / 1074's 50 synthetic candles: cycle 2 on cycle 1's CTS candle 46, BOS 47, confirmation 47 —
  both modes raise). The full suite passed with the raise before the new tests (1359): no fixture fed an
  establishment before its BOS. Mutant (the raise → the old silent skip): 2 / 2 relevant pins fail. No separate
  landing review (a ~10-line change): folded into S3's.
