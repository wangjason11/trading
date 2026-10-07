# Plan I — The cross-cycle fib revamp: one logic for main and sub (Week 9 Part 1 item 2)

**Status: SPEC AGREED — rounds 1–5 ANSWERED (2026-10-07; §6, last R5-1…R5-3): no open question. Unique subs numbered
at their lifecycle START (R4-2); `cross_fib_limit_source` kept; the read rules R4-5 (inclusive) / R4-6 (an ended x−1)
accepted, an ended x−1 yielding its FROZEN thresholds (R5-3); R4-7 withdrawn — the user's ordering argument holds in the
code (R5-1). Session 1 CLOSED 2026-10-07 (docs-only commit); NEXT: session 2 — the implementation plan (§8).** Session 1 of 3 (PLAN_H §6 LR3: the user's rules → this spec;
then the implementation plan from a fresh code audit + a cold review; then the build). Living document: every
question, answer and decision is recorded here as it happens (§5 open questions, §6 decisions); measurements in §3.
Register line: `IDEA_PARKING_LOT.md` §D.1 item 2. No code changes in this session.

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
   | main: reversal | "the most recent bos_threshold that a valid pattern broke out of that triggered the reversal" → the APPLIED reversal's `bos_frozen` (`REVERSAL_CANDIDATE.meta["bos_frozen"]` of the reversing sid; the MS barrier the reversal pattern broke — an earlier failed close-break `rv_anchor_failed` moved the threshold, not this) |
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
     built whole at its start, and the clamp keeps every fib inactive before it.

## 2. What changes (round-1 reading, from the census §3 — nothing built; still valid after rounds 2–5: the x−1 rule gives the same `P` on the window, §3.3)

- **Subs:** a cap where there was none. On the window 6 of today's 8 sub cross records become singles (conf sub 0
  cyc 1–4: no `P`; sub 2 cyc 1: `M = 0`; sub 3 cyc 2: `M = 1`); sub 7's two stay (Q1 (a): `P` 0.57827, `M = 2`).
- **Main:** sid 1's `P` = the reversal's `bos_frozen` 0.5736 = today's KL-zone `P_rev` → `M = 2`, as today. The
  changes come from I2 / I5 / Q4–Q6 / Q8: two NEW H1 cycle-0 single fibs — sid 0 cyc 0 (BOS 96 → CTS 430; live: a
  new fib + IP POI record, and through MS's sd-proximity possibly (0,0)'s confirmation, today by pullback at 439) and
  sid 1 cyc 0 (689 → 710; collapsed — it ends before the reversal applies at 902 — so hidden); the pre-established
  cross on main (predicted none on the window, §3.1); the (1,1) cross's meta (Q8). Measured by variant replays in the
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
  plan's variant replay).

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
    `prev_sub_bos_threshold` (R2-1), `prev_sub_starting_price` (R2-2), `prev_sub_latest_cts` (R2-b). WHAT IT SHOWS:
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
    §3.2) — so the limit never revokes a cross already made.
  - Where they live (for the plan session): on the structure (the pooled sub / the main sid's tracker state) and
    copied onto every fib record's meta (CSV + hover) — new, additive meta keys. Optional, plan session:
    `cross_fib_limit_from_sub_id` (the sub that supplied `P` — x−1, or the reversing sub).
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

## 8. Handoff to session 2 — the implementation plan (from a fresh code audit + a cold review)

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
