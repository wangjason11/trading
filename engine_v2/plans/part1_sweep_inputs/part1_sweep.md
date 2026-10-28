# Week 9 Part 1 — SESSION ORDER item 10: every parked / deferred item (the sweep, 2026-10-28)

The user's request (verbatim, register `IDEA_PARKING_LOT.md` §D SESSION ORDER item 10): "I want to start with a sweep
of all parked and deferred items we've noted across all our documentation. I want them listed out with a short
description of what they are. Please exclude any items that we've already agreed would be covered in Part 2 & 3 of
Week 9. After you provide the list, I will decide if there are anything else I want to cover before moving forward to
Part 2 of Week 9. Part 2 is an important delineation from Part 1, so I want to make sure our overall MS is totally
solid before moving on to Part 2."

## How it was built

- **Spine:** the register, read in full (buckets A–I + "Recently closed"), HEAD `132f5c2`, suite 2173 passed.
- **Cross-check:** `sweep_grep.py` (this folder) — deferral wording over 84 repo `.md` (minus `.venv` / `artifacts`
  / the legacy side projects), the 4 project skills, all 59 memory files and 494 `engine_v2` `.py` files: **1,085
  hits in 121 files** (`sweep_grep_output.txt`; 124 of them the register itself). Every hit outside the register was
  read in its heading's context; the uncertain ones were checked against the code or S_W9d — never against a note.
- **On-window facts quoted below:** `sweep_checks.py` → `sweep_checks_output.txt` (S_W9d
  `20261028_182318_7c9477e` + its `run.log`).
- **Code TODO / FIXME / XXX:** none. The 11 code comment hits are explanations, not open work.
- **Excluded (agreed for Week 9 Parts 2 / 3 — register §D.3 + SYLLABUS Week 9):** zone size at entry and stop loss;
  the entry confirmation candle patterns; the candle-direction / rejection confirmation at a zone touch; zone strength
  scoring (+ its ATR-percentile idea); the M5 layer (+ the deeper-path encoding PART4 §2 "Open (deferred)", M5
  nesting PART4 §17.12); Part 2's own plan (EntryPlanner, the Stop/TP builder, RR ≥ 3, position sizing, the entry /
  stop / TP plot).

**Legend.** ★ = touches market-structure or zone correctness (what Part 2 will read). Status: **OPEN** (to do) ·
**DECIDE** (the user's call first) · **TRIGGER** (waits for a condition) · **PARKED** (by the user) · **Week 10** ·
**SUPERSEDED / CONFIRM** (looks done — close?). "NEW" = was not in the register; its line was added by this sweep.

---

## 1. Market structure + the start probes

**1.1 ★ The probe's BOS_0 zone reads past its search bound ("L5b")** — NEW. At a reset candidate the probe builds an
ad-hoc BOS_0 zone (`unified_probe._bos0_inner_at_start` → `build_ad_hoc_bos0_reference_zone` →
`identify_base_pattern`) that reads up to 5 candles past `probe_end_idx`. Not live look-ahead (those candles exist when
the trigger fires), but the probe's stated bound is not its real bound. Found 2026-09-19 by Plan A ("deliberately
open"): "closing it can move the H1 sid-1 start through the main reversal probe, so it needs its own decision and
`/compare`". Never measured. **DECIDE.** LANDMINES "Probe `end_idx` Is the Supreme Bound"; PLAN_A §8.

**1.2 ★ Bounded MS runs see the full-frame imbalance list** — NEW. `df.attrs["imbalances"]` is built on the whole
frame, so a run that stops at candle B still sees a gap whose third candle is B+1. Narrowed by Plan F (2026-09-24):
only a degeneracy mismatch between the prefix and the merged gap can reach an output — 0 cases on the reference data;
sub builds are immune. "If the property test ever trips on it, record and decide." **TRIGGER.** LANDMINES "Bounded MS
Runs Must Not Read Past `end_idx`" (Known residual); MEMORY.md "Residuals: … L5b open".

**1.3 ★ MS's POI inner is a snapshot** — MS's sd-proximity CTS confirmation reads POI inners refreshed only at
CTS_ESTABLISHED / CTS_UPDATED (and only until the cycle's CTS confirms). An imbalance fill between two refreshes that
disqualifies the IC (or changes the pick) is not seen until the next refresh. The spec: "In practice the deviation is
small" — never measured on today's code. Filed under Week 10 (full per-candle recompute is expensive), but it decides
backtest CTS confirmations now. **Week 10 / DECIDE (measure?).** MARKET_STRUCTURE_SPEC "POI inner refresh" (~:440–457);
register §F.

**1.4 ★ Offline pattern selection can establish a cycle later than a live engine would** — anchor-first selection
tests a continuous SUCCESS before an earlier 2-candle SUCCESS, and anchors inside a back-fill are never
pattern-tested, so a cycle can be established LATER than candle-by-candle (never earlier). Code-read only
(2026-09-28), never measured — measuring needs a live-like run. **Week 10.** Memory `project_zones_timing_audit_20260922.md`
latent (b); register §F (with moment-order processing, PLAN_E Q3).

**1.5 ★ `REVERSAL_CANDIDATE`'s `pattern` meta is always `"?"`** — NEW. `_schedule_reversal_from_anchor` reads
`ev_r.pat` / `ev_r.pattern`, but the pattern's name is `ev_r.name` (`market_structure.py:811`, `:848`). On S_W9d: all
6 candidates (H1 1, confluence 5) say `'?'`; the `[RV_SCHEDULE]` / `[RV_APPLY]` log lines too. The STATE_CHANGED →
reversal `meta["pat"]` is right. Meta only — no decision reads it. Found 2026-09-19: "a one-line fix (`ev_r.name`) for
a later commit — it changes event meta, so it gets its own `/compare`". **OPEN.** GOTCHAS "`REVERSAL_CANDIDATE.meta
["pattern"]` Is Always `"?"`".

**1.6 ★ A `REVERSAL_CANDIDATE` can straddle a sub's cap** — `knowable_at_idx` keys the candidate on its pattern candle,
not its `apply_idx`, so a sub capped for another reason could keep a candidate that applies past the cap. 0 on the
window (measured 2026-09-30; not re-measured since). **TRIGGER.** PART4 §17.12; register §E.

**1.7 ★ B5 — a BOS confirmed on a reversal's own apply candle** — both reversal builders take the latest BOS_CONFIRMED
with moment ≤ the reversal (inclusive); such a BOS's threshold would not be the frozen barrier. Latent, 0 on the
window. **PARKED** by the user 2026-10-28: "I think this case is not possible, so let's park for now". PLAN_K §11.2
B5; register §D.2.

**1.8 ★ A double RANGE_STARTED from the offline finalize** — fixture-only, 0 on the reference window. **PARKED** by the
user (2026-09-28). Memory `project_zones_timing_audit_20260922.md`; register §E.

**1.9 The probe's iteration cap is H1's on every TF** — NEW. `unified_probe` gives up at `max_iterations = 10` (every
iteration reset → `pending`), calibrated for H1; the M15 probes use the same value. On S_W9d: 24 probe runs, at most 4
iterations (12 × 1, 7 × 2, 3 × 3, 2 × 4), no `max_iterations` finalize. PART4 §15 "Open Items": "Revisit when wiring
up the TF-keyed thresholds". **TRIGGER** (a probe finalizes `max_iterations`, or M5 probes).

**1.10 Two verification gaps around the first_confluence probe** — (a) no end-to-end pin that an unblocked probe whose
cycle 0 confirms by proximity reaches its 2nd CTS where its sub does (no fixture found, 2026-10-27); (b) the
probe/sub consistency tripwire misses a probe confirming cycle 0 BEFORE its sub. **TRIGGER** (both). PLAN_T §5;
register §E.

## 2. Zones — fib, IC / POI, imbalance, candle types

**2.1 ★ Re-ask a failed fib decision when the gap forms (Plan F §7)** — FibTracker decides only at CTS events; when the
imbalance a fib needs completes (its third candle closes) one candle AFTER the event, the fib is dropped instead of
created a candle later. "User-accepted" follow-up (2026-09-24). Downstream: no fib → no POI → no POI proximity
trigger / confirmation. Not measured on today's code. **TRIGGER** (a chart review shows a dropped fib) / parked.
PLAN_F §7; IMBALANCE_FILL_SEMANTICS "When to revisit"; register §E. Same register line, display / debug only (all
still in the code): the fill state `armed_idx` / `confirmed_fill_idx` written onto imbalance instances shared across
projections (the debug export shows the last writer); the chart's POI hover picks a POI by nearest inner price (can
show an inactive twin); a single-candle active stretch is not drawn (`sx1 <= sx0`, 3 sites). **SUPERSEDED** in that
line: "the MS vs FibTracker activation divergence" — MS reads FibTracker's decision since PLAN_I S5 (2026-10-08).

**2.2 ★ Step 4 of IC identification — the lower-TF fallback** — when no IC candidate is left (`no_ic_candidate`,
`all_ic_candidates_pinbar`), rerun Steps 1–3 on the lower TF; today no IC → no POI. **KEEP DEFERRED** (the user
2026-10-28): decide once Part 2's entries show whether the records without a POI matter; "We may also ultimately
choose to skip Step 4 entirely" (2026-10-09). The 7-of-26 no-POI count is from PLAN_P (2026-10-13) — re-measure first.
The unwired V30 / V60 / V90 selection code stays ("Keep it for now", PLAN_H ICR-5). Register §D.2.

**2.3 ★ The pinbar body floor is still provisional** — NEW as its own line (was only inside the closed D.1 item 4).
The rolling floor's 90 candles / p90 are a first choice on NZD_USD; "the reclassify-to-pinbar approach itself may be
revisited (e.g. a separate 'no-conviction' type, or a softer demotion)"; "watch the interaction with
`one_maru_opposite` / `one_maru_continuous` pattern alt-paths". Candle types feed the patterns and so the structure.
**TRIGGER** (strategy optimization, Week 10). CANDLE_BODY_FLOOR_NOTES "Open / may-revisit"; PLAN_U §7.

**2.4 Fib storage unification** — `_cross_cycle_data` → versioned `_fibs`; "DEFERRED, not closed (user 2026-05-27): may
revisit". A refactor meant to be byte-identical. **PARKED.** FIB_LIFECYCLE_SPEC §13; register §G.

**2.5 Two names for a cycle's end** — the fib says `new_cycle`, the cycle table (KL / POI) says `next_cycle`; both show
as `end_reason=` on one chart. A rename is an exported-meta migration. Flagged 2026-10-23, never asked. **DECIDE.**
Register §G.

## 3. Subs, the pool, multi-TF plumbing

**3.1 ★ Would-have-been subs for collapsed parent cycles** — the pool builds no sub on a collapsed parent cycle (on
the window H1 sid 1 cycles 0 / 1 → the 4 unresolved triggers: first_confluence 2815 / 2995, first_counter 3283,
subsequent_confluence 3487), so the opp_sd proximity overlay reads parent zones only there. The user 2026-10-19: "OK we
can go with parent zones only for now and consider subs later". **PARKED.** Register §G; `qa_opp_sd_zones_inputs/
qa_findings.md` rounds 4–6.

**3.2 Audit positional-index df columns that survive the sub slice** — **TRIGGER** (adding any df column whose value is
a positional index). Memory `project_is_range_pollution_fix.md`; register §E.

**3.3 The mirror initializes new columns with `pd.NA`** instead of the per-column defaults — **TRIGGER** (it bites; the
slice-copy drop is the safeguard). LANDMINES "Slice Copies Inherit Mirrored Structure Cols"; register §E.

**3.4 PART4 §15 "Validation strategy"** — NEW. "How to systematically validate per-entity isolation, registry
correctness, event-bus ordering, and recursive triggering. Likely needs new test infrastructure." Looks
**SUPERSEDED**: 2,173 tests incl. the pool / sweep / registry / per-entity suites, the mutant runs and the `/compare`
protocol. **CONFIRM** (close it).

**3.5 Split `PART4_REFACTOR_SPEC.md` into canonical spec files** — docs only. **PARKED** as optional by the user
2026-09-30. Register §G.

## 4. WVMI, wave candles

**4.1 Per-LP WVMI rows for collapsed records** — "for now collapsed records don't need to get per-LP rows. I may
change my mind later" (2026-10-26). **PARKED.** Register §G.

**4.2 The batch `update_temporary_lp` pass does nothing in batch mode** — still called (`orchestrator.py:204`); matters
only live. **Week 10.** Register §F.

## 5. Charts and hovers

**5.1 The charting audit's unverified list** — each would change a rendered figure: the H1 chart draws a collapsed
cycle's fib; proximity markers on collapsed-cycle zones; the opt-in M15.counter KL overlay on H1 has no collapsed
skip; the H1 chart drops the wave-candle `dash`; wave verticals at forming candles take the live style; the H1 chart
vs H1 overlay filter quirks; a sub whose confirmed points are all forming draws no live segment; PART4 §16.7 zone text
labels (specified, never rendered — build or drop). **OPEN** ("the user's call when touched"). Memory
`project_charting_audit_20260921.md`; register §A.

**5.2 The BOS / CTS dot hover quirks (5)** — NEW as its own line. A PB dot's `cycle_id` is a placeholder 0; a sub PB is
a pullback or a replacement break, both `kind=PB`; `idx` = the anchor, the confirmation moment / method / pattern not
shown and `phase` judged at the anchor; `time_1H` stored but never shown, `layer=recent` constant; the M15 overlay
labels an unconfirmed CTS `CTS` where H1 says `CTS (established|updated)`. The user 2026-10-21: "Let's come back to it
later to address them in more detail". **PARKED.** `qa_chart_edits_inputs/user_spec_20261022_dot_hover.md` "Quirks".

**5.3 The hovers not yet on the hover conventions** — NEW. The OHLC candle (but its body-floor line), volume / EMA,
the range top / bottom lines, the reversal candidate, the state / `bo` labels, the structure-pattern markers (no
`TF=` on H1). **OPEN.** CHARTING_SPEC "Hover label conventions".

**5.4 The fib hover masks candle hovers** — each fib's invisible hover line (one point per candle, BOS → CTS anchor)
wins Plotly's "closest" hover over the candles under it. **PARKED** ("come back … overall chart revamp"). Register §G.

**5.5 The proximity markers' `zone_confirmed_idx` means two things** — a KL marker shows the lifecycle-clamped
confirmation, a POI marker the unclamped activation floor. **PARKED.** CHARTING_SPEC §12; register §G.

**5.6 Uniform 3-tier chart opacity** — one `opacity_for(sid, cycle)` helper (cycle live / sid live / sid ended) for
every element. Its own session. **PARKED.** FIB_LIFECYCLE_SPEC §9.4 + §13; register §G.

**5.7 The stepped "current proximity reference" line on the H1 chart** — the user 2026-10-20: "Decide later". A sample
renders from `qa_opp_sd_zones_inputs/build/prox_ref_sample.py`. **PARKED.** Register §G.

**5.8 M15 POIs / fibs / confluence zones on the H1 chart** — NEW. "Not yet rendered on H1 (future work)". **OPEN.**
CHARTING_SPEC §11.

**5.9 A breadcrumb header with cross-chart navigation** — "not in v1". **PARKED.** PART4 §16.9; register §G.

## 6. Naming / hygiene

**6.1 Plan E's optional big renames** — NEW. `probe_end_idx`, `creation_event_idx`, `enforce_cts0_new_extreme`, the test
constant `CONF` — "Optional big renames, deferred" (PLAN_E_inputs §2.8, 2026-09-24); all still in the code. **PARKED.**

## 7. Week 10 — live readiness (scheduled there; listed for completeness)

Register §F + §H + §E "Performance": incremental `is_filled`; incremental imbalance detection; per-candle POI inners in
MS (= 1.3); the Phase 3 per-candle dual-lens driver + live pool GC; ad-hoc zone wait-for-candles; moment-order event
processing + live-mode pattern selection (= 1.4); a live / incremental probe; the WVMI batch pass (= 4.2); `armed_idx`
chart surfacing; consumer-specific fill predicates; performance (entity-direct MS compute without the slice,
same-TF feature sharing by reference, parallelism).

---

## Stale notes found by the sweep (no decision needed — fixed in the docs with this sweep)

- PART4 §17.12's last bullet ("Whether subs should receive the derived CTS KL zone … the branch is dead for subs
  today") — DONE by PLAN_W 9c-2 (2026-10-28).
- Register §E "Plan F §7 follow-ups" still listed "the MS vs FibTracker activation divergence (measure first)" —
  retired by PLAN_I S5.
- Memory `project_zones_timing_audit_20260922.md`: "the dead `scenario_context` … Open" (gone from the code) and "the
  commented old `_emit_cts_established` / `_emit_cts_updated` bodies … remain" (no commented-out `def` left in
  `market_structure.py`).
- Memory `project_cycle_lifecycle_parent_cycle_floor.md`: "Chart-or-remove the outline-only sub zones — decide later" —
  decided since (collapsed zones are hidden; CHARTING_SPEC "Recent vs prior").
- Memory `project_plan_k_ic_reference.md`: "Still open from before: sub 3 cyc 3 has no POI record" — closed by PLAN_W
  9b.

Checked and already closed (no change): every PLAN_E_inputs "Still open" question (answered through E1–E5); PLAN_I's
F14 / F16 / F17 / F19 and R4-7 (fixed / withdrawn); PLAN_H's "ranking rule 4 over two candidates sharing one gap"
(ICR-1); PLAN_R's round-6 opens (crossed references = narrow, the renames); CROSS_CYCLE_FIB_SPEC §9 (all reconciled);
the `bos_threshold` reset bug (fixed 2026-06-01); the probe's `DEFAULT_PROXIMITY_PIPS` tightening (superseded — 8 / 5
/ 3 since 2026-09-30).

---

## The user's pick (2026-10-28, AskUserQuestion — the answers verbatim)

- MS / zone items: **"Measure 3 MS risks (Recommended)"** (1.1 the probe's BOS_0 read past its bound, 1.3 MS's POI-inner
  snapshot vs as-of, 2.1 fibs dropped when a gap completes a candle late — measured on the reference window + other
  windows, no code change; fixes decided after, one plan / `/compare` each), **"Fix the '?' pattern meta
  (Recommended)"** (1.5, its own `/compare`), **"Re-check parked latent cases"** (1.2, 1.6, 1.7, 1.8 counted on other
  windows). Not picked: the POI-less re-count (2.2).
- Charts / hovers / hygiene: "The overall charts look good for now. We can consider any remaining charting items
  mentioned in the documentation to be "long term" parked. These we may not come back to unless I specifically mention
  something about doing another detailed sweep." → group 5 LONG-TERM PARKED (register §A / §G marked).
- PART4 §15 "Validation strategy": **"Close it (Recommended)"** → closed (PART4 §15, register "Recently closed").
- The sweep docs: **"Commit now (Recommended)"**.
