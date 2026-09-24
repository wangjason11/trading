# Plan F inputs — imbalance c3 knowability: consumer audit + live measurements (2026-09-23)

## 0. What this is

**This is the INPUT for the c3-knowability plan. It is not the plan.** It is the persisted output of the session's
`/prepare` (memory `feedback_persist_prepare_audits.md`): every imbalance consumer's as-of read, classified and
adversarially verified, plus live measurements of every call site on the reference window. The scope decision goes to
the user; the plan (predictions, tests, docs) is written after it and cold-reviewed before any code.

- **State:** HEAD `754a642` (docs-only on top of Plan D `0a4eadc`). `/compare` baseline = save
  **`20260923_172626_0a4eadc`**. Line numbers are HEAD `754a642` coordinates.
- **Sources:** (1) a read-only audit workflow — 7 group readers (133 site records) + a completeness critic (9 more) +
  one adversarial verifier per group + a synthesizer (raw records: `audit_workflow_records.json`); (2) four
  measurement scripts in this folder, each run as a full replay from the repo root (every run passed the `/compare`
  M15 fetch gate: `Fetched 4228 M15 candles … in 5 chunks`, no `[data_bridge] ERROR`).

## 1. The defect, stated precisely

`compute_imbalance` flags c2 (the MIDDLE candle). An `ImbalanceInstance(start_idx, end_idx)` is a run of c2s; its gap
exists only once a c3 closes. Two leaks:

1. **POI activation sweep** — `poi_zones.py:920` `enter_idx = max(inst.start_idx, first_active)` enters the instance
   into the unfilled set at c2, one candle before it exists.
2. **The shared primitive** — `has_unfilled_imbalance` (`imbalance.py:159-166`) calls `inst.is_filled(check_to_idx)`,
   which returns False (= UNFILLED) on an empty scan range (`types.py:223-224`, `end_idx >= check_to_idx`). So any
   instance overlapping the window whose c3 has not closed by `check_to_idx` counts as an unfilled imbalance.

**Knowability rules considered.** R1 (existence): an instance is visible as of K iff `start_idx < K` (its first c3
has closed); exact form overlaps on the knowable prefix `[start_idx, min(end_idx, K-1)]`. R2 (final bounds): visible
iff `end_idx < K`. R1 == R2 for single-candle instances; they differ on merged runs (43/167 H1 instances, 174/622 M15).
**R1 is live-faithful**: a live engine sees the prefix from `start_idx+1`, and the prefix's only scanned candle is its
own c3, which cannot arm its own gap (for `fill_threshold > 0`), so "prefix unfilled" == "full instance unfilled" for
every `K` in `(start_idx, end_idx]`. Caveat: a prefix and its merged run can differ in degeneracy (`gap_size <= 0`
→ `is_filled` True) — 0 degenerate prefixes on this data (H1 0/218, M15 0/842).

**The cutoff K must be the evaluation MOMENT, not `check_to_idx`.** Many callers pass an anchor as `check_to_idx`
(the fill as-of; Plan E E3 territory). Measured: 2 pattern-path calls had `check_to = cts anchor = apply_idx - 1`
with the instance at c2 = the anchor — knowable at the moment (its c3 IS the apply candle); a cut keyed on
`check_to_idx` would wrongly drop it. Lag-1 CTS_ESTABLISHED cycles (1223/1224, 2828/2829) are the same shape.

## 2. Measurements (reference window)

### 2.1 Shadow — every call wrapped, original result returned (`shadow_measure.py`)
One replay; each `has_unfilled_imbalance` call and each POI-sweep call recomputed under R1 and R2; outputs
byte-identical to the current code. Flipped calls per site (calls / flips):

| Site | Calls | Flips | Notes |
|---|---|---|---|
| POI sweep (`_compute_poi_activation_history`) | 37 | 3 | H1 IC 865, 860 (sid 1 cyc 2); M15 sub IC 4048 |
| `poi_zones.py:229` via MS `compute_poi_inners_for_cycle` | ~27.7k | 1080 | raw path 440 R1 + 479 R2-only; pattern path 15 R1 + 2 anchor-artifact + 150 R2-only |
| `poi_zones.py:229` via `derive_poi_zones` | ~3.2k | 47 | 39 R1 (all counter sub n=657 cyc-1 fib, check_to 235) + 8 R2-only |
| `market_structure.py:2050` cycle-0 snapshot | 350 | 14 | 4 R1 + 10 R2-only |
| `fib_tracker.py:536` (CTS_ESTABLISHED) | 27 | 3 | 1 R1 (M15 EST@53, lag 0) + 2 R2-only |
| `fib_tracker.py:1183` (cross_cycle cyc-0 activation on update) | 1 | 1 | R1, raw CTS_UPDATED@61 |
| `fib_tracker.py:1462` (`_update_fib_cts`) | 246 | 4 | all R2-only |
| `cross_cycle_fib.py:117` | 137 | 2 | 1 R1 (CTS_THRESHOLD_UPDATED@235) + 1 R2-only |
| `cross_cycle_fib.py:149` / `:156`, `fib_tracker.py:1318/1527/1535/1540` | 8 / 162 / 6 / 6 / 6 / 6 | 0 | |
| `fib_tracker.py:1436/1445/1451/1571` | 0 calls | — | dead / unreached |
| `get_unfilled_imbalances` | 0 calls | — | import at `fib_tracker.py:25` unused |

"R2-only" = a merged run mid-formation (R1 sees it, as a live engine would; R2 drops it) → R2 over-excludes.

### 2.2 Cascade — the rule APPLIED per scope, 24 CSVs diffed vs the baseline (`variant_run.py` + `diff_variant.py`)
K = the evaluation moment: MS pattern path `_apply_pattern_at_apply_idx.apply_idx`, MS raw path
`_maybe_update_cts_pre_confirm.i`; downstream CTS_ESTABLISHED `meta["confirmed_at"]`, any other event `ev.idx`;
`derive_poi_zones` K = `check_to_idx`.

| Variant | Changed calls | CSVs identical | Delta |
|---|---|---|---|
| **sweep** (enter at `start_idx+1`) | — | 21/24 | exactly 4 POI rows: H1 IC 865 + 860 `confirmed_idx` 997→998 (re-activations 953→954, 997→998); M15 IC 4048 4118→4119 in conf + counter |
| **ms** (in-flight resolver + cycle-0 snapshot) | 459 | **24/24** | none |
| **fib** (downstream FibTracker + cross_cycle_fib) | 3 | 21/24 | conf fib_lifecycle 2 rows (sub2 cyc0 `activated_at` 53→54 + `activated_on: update`; sub3 cyc0 61→62); counter fib_lifecycle 9→8 rows (sub3 61→62; the sub5 cyc1 cross fib is never created; sub5 cyc0 end 3806 `new_cycle` → 4083 `same_dir_replacement`); counter POIs 13→12 (IC 3654 — the parked never-established fallback POI) |
| **poiid** (`derive_poi_zones` IC identification) | 39 | **24/24** | none |
| sweep+ms+fib | — | 19/24 | exact UNION of the single-scope deltas (no interaction) |
| sweep+ms+fib+poiid | — | 19/24 | byte-identical to sweep+ms+fib |

### 2.3 Why the MS in-flight path is 0-delta (`inners_shadow.py`)
Of 451 resolver calls, 3 change the inner PRICES (all raw path, all cycle 0: H1 sid0 i=163 −3 inners, i=171 −2;
M15 sub n=1639 i=61 −2). The run has 3 sd-proximity POI fires, none on a lost inner. Structurally: the only reader of
`poi_inners_for_cycle` (`market_structure.py:1957`) is gated `i > st.cts.idx` and `cts_cycle_id > 0`
(`:1975-1981`), so every instance counted at refresh time has its c3 closed by the candle it is USED on — a
compute-time look-ahead, not a decision-time one (audit row 9-12, Q1).

---

# Audit synthesis (workflow output; reconciled with §2 — where it cites `[INBOX:n]` the numbers are §2's)

**Basis.** This merges the reader and verifier records from G1 to G7 and the critic. It also read (did not modify) this session's `memory/_INBOX.md`, whose shadow and cascade measurements it cites as [INBOX:line] — those numbers are §2 above. I ran no measurements myself. To settle disputes I re-read the code, and this confirmed the following:

- the sweep enter at `poi_zones.py:920`;
- the primitive at `imbalance.py:159-166`, with the empty-scan branch at `types.py:223-224`;
- the proximity gate at `market_structure.py:1975-1981`, and that `:1957` is the only reader of `poi_inners_for_cycle`;
- the `select_fib_anchor_for_cycle` wiring at `fib_tracker.py:155-193`;
- the fib sites at `fib_tracker.py:536`, `:763`, `:1183`, `:1318-1360` and `:1421-1595`, and the threshold path at `:2061-2116`;
- `cross_cycle_fib.py:114-164`;
- the moment emitter at `market_structure.py:1493` (`confirmed_at=int(apply_idx)`);
- the Rule-3 latches at `zone_proximity.py:385-455`;
- the 16 production call sites of `has_unfilled_imbalance`, found by grep: fib_tracker 11, cross_cycle_fib 3, market_structure 1, poi_zones 1. The POI sweep does not go through the wrapper.

## Audit table

Documentation and test sites are listed in §3. "Disputed" in the classification column means readers and verifiers disagreed and I ruled; §5 gives the reasoning.

| # | Site (file:line, function) | Question | Window / check_to_idx | As-of kind | Unknowable instance reachable? | Classification | Proposed fix / note |
|---|---|---|---|---|---|---|---|
| 1 | zones/poi_zones.py:920, `_compute_poi_activation_history` (enter) | From which candle does a same-direction instance count toward cond3? | `enter_idx = max(inst.start_idx, first_active)`, with first_active set at :839-841. As-of is the sweep candle t in [first_active, scan_end] | processing_candle | Yes. The instance enters at c2 whenever start_idx >= first_active. Measured last session, reproduced by G1: H1 sid1 cyc2 IC 865/860 re-activations 953→954 and 997→998 (confirmed_idx 997→998); M15 sub7 IC 4048 4118→4119 in both lenses [INBOX:33-34] | **fix** | Use `enter_idx = max(inst.start_idx + 1, first_active)` (R1). Also update the comments/docstrings at :516-518, :809-812, :817-819 and :924-925. Write the tests first (§3). |
| 2 | poi_zones.py:912-918, relevant_imbalances filter | Which instances can ever enter the sweep? | sd direction, gap>0, end_idx > ic_idx, start_idx <= scan_end | processing_candle | Only start == scan_end can pass the filter and then never enter; :921 drops it once enter = start+1 | equivalent | None required. `end > ic` is equivalent to `start > ic`, because the IC is a −sd candle (:218-220). If :917 is changed, mirror it in the debug dump at :645. |
| 3 | poi_zones.py:928-933, committed-filled skip and leave | Already filled at enter? At which candle does it leave? | Cached confirmed_fill_idx compared with enter_idx | processing_candle | Present but inert. confirmed >= armed >= end+2, because the last c3 sets the gap edge and cannot arm its own gap. This needs fill_threshold > 0 (hardwired 0.70 at orchestrator.py:307). Measured min = 2 on H1 (167 instances) and M15 (622) | equivalent | None. |
| 4 | poi_zones.py:657-776 (scan starts at :733), `_compute_fill_idx_cache` | At which candles do stroke 1 and stroke 2 fire? | Scan from end_idx+1 to the end of df | not_applicable | No. The scan starts at the last c3 (the R2 point), and nothing in [start+1, end+1] can arm | already_bounded (disputed) | None. The `inst.meta` stash at :773-774 is write_only, but the tuple itself drives :923 and :928-933. |
| 5 | poi_zones.py:547-570, history fold (confirmed_idx / current_versions / status) | Last activation, live variants, status | Fold over the finished history | final_state_retrospective | Inherited from row 1 | equivalent (outputs move with row 1) | None. Edge case: an empty history is labelled 'inactive' before the 'ended' check (:567). |
| 6 | poi_zones.py:606-652, POI_LIFECYCLE_DEBUG dump | Debug print | Mirrors :912-918 | not_applicable | Not a decision | write_only | Mirror any change made at :917. |
| 7 | poi_zones.py:229, `find_ic_candidates` via `derive_poi_zones` :457 | IC cond3: is an sd instance after the candidate unfilled as of CTS? | Window [idx+1, cts_idx]; check_to = fib_state.cts_idx (:204, :233) | anchor (final fib CTS) | Only when start == cts_idx. None on H1. M15 counter sub5 cyc1 has 39 candidates affected but 0 POI delta; the "poiid only" variant is 24/24 identical [INBOX:41-42] | defer_to_plan_e | No c3 filter here. This is retrospective IC identification, and the sweep enforces the timing. It is PLAN_E §2.4 item 8 (PLAN_E_inputs.md:639, :673-674). If the primitive gets a keyword, this call does not pass it. |
| 8 | poi_zones.py:448 (derive gate) and :198 (`find_ic_candidates` gate) | Which fibs feed POIs? | Final FibState active / locked / end_idx | final_state_retrospective | Inherited from fib_tracker. A flipped or never-created fib removes all of its POIs | equivalent | None. Exposure: the 3 M15 records that pass on `active` alone (conf sub7 cyc2; counter sub5 cyc1; counter sub7 cyc2). :198 is looser than :448, and on the in-flight caller it always passes (FibState is built active at :1084-1094). |
| 9 | structure/market_structure.py:1944-1987, `_check_proximity_at_candle` / `_maybe_confirm_cts_via_proximity`. This is the only reader of `st.poi_inners_for_cycle` (:1957) | sd-proximity CTS confirmation at candle i | Snapshot read at i, gated on `i > st.cts.idx` and `cts_cycle_id > 0` (:1976-1980) | processing_candle | No. Every snapshot window ends at cts_idx <= i−1, so every admitted instance's c3 has closed by i | already_bounded (disputed) | None. Record this consumer-gate bound in MARKET_STRUCTURE_SPEC.md:203-212 so nobody adds a filter at refresh time. |
| 10 | poi_zones.py:229 via `compute_poi_inners_for_cycle` :1098, the MS resolver (refreshed at market_structure.py:1561, :1718, :1724, then :2003) | Is the candle an IC candidate for the in-flight POI inners? | Window [idx+1, st.cts.idx]; check_to = st.cts.idx (:2007) | mixed (raw path: the processing candle; pattern path: the anchor, which is <= apply) | At refresh time, yes: start == cts, 1080 flipped calls [INBOX:22-24]. At the candle where it is used, no (row 9) | already_bounded (disputed; the G5 verifier said fix) | No filter. The "MS in-flight only" variant is 24/24 byte-identical [INBOX:35]; only 3 of 451 calls change inner prices, all unconsumed [INBOX:45-48]. See Q1. |
| 11 | cross_cycle_fib.py:117 via poi_zones.py:1052 then fib_tracker.py:169 (in-flight Scenario-2 cond1) | Does cycle 1 have its own unfilled imbalance in [BOS_1, CTS_1]? | current_candle = st.cts.idx | mixed | At refresh, yes (H1 751 and 760 in scope, 0 flips). At use, no (row 9) | already_bounded (disputed) | No filter. Any new keyword on `select_fib_anchor_for_cycle` must default to None so this caller is unchanged. Note the exception under LANDMINES.md:188-223. |
| 12 | market_structure.py:2050, `_update_cycle0_data` (MS copy of cond2) | Is cycle 0 unfilled in [BOS_0, CTS_0] as of CTS_0? | Window end == check_to == st.cts.idx | mixed | At write time, yes: flips at H1 sid0 163/171 and M15 slice-local 53/61, all sid 0 and never consumed (fib_tracker.py:155). At use (a cycle-1 refresh, read at i > CTS_1), no | already_bounded (disputed) | No change in the c3 commit. Today cond3 implies cond2 (cross_cycle_fib.py:149-154: same window, BOS_1 > CTS_0, fills monotone), so cond2 never binds. cond2 semantics is Q2, decided together with row 17b. |
| 13 | cross_cycle_fib.py:149, snapshot walk (cond3 @BOS_1); callers fib_tracker.py:880 and poi_zones.py:1052 | Did BOS_1 fill cycle 0? | [BOS_0, CTS_0] as of own_imb_start = BOS_1 | anchor | No: BOS_1 > CTS_0 strictly (per the G3/G5 verifiers: market_structure.py:1004-1014, :1710, :2234-2238) | already_bounded | None. |
| 14 | structure/unified_probe.py:533-544, Phase-2 bounded MS on the full-frame `attrs["imbalances"]` (LANDMINES.md:2112-2119 residual) | Can a bounded run use an instance created by a candle past B? | Windows end at cts <= P <= B | mixed | An instance with start >= B is counted only when cts == P == B, and read only past B. Phase 2 runs sid 0, so the path is intra (fib_tracker.py:155) | equivalent | No code change. Rewrite the residual wording at LANDMINES.md:2112-2119 and MARKET_STRUCTURE_SPEC.md:326-332: the existence half is unobservable; the degenerate-gap corner remains. |
| 15 | multitf/entity_df_mutation.py:1060-1067 and :1103 (`_build_geometry`), and multitf/data_bridge.py:82 (the instance producers) | Which instances exist in the frame? | The sub slice runs to run_cap_abs = len(m15_df)−1, the data edge (:1138); data_bridge.py:82 covers the full M15 frame | not_applicable | Producers only. The last row is never flagged (imbalance.py:55) | already_bounded (only at the data edge) | None. This is not protection at the sub's cap: the sub sweep is bounded only by scan_end (poi_zones.py:495), so row 1's leak applies at sub caps too. |
| 16 | zones/fib_tracker.py:536, `on_cts_established` (feeds activations at :590, :651, :719, :773, :833, :853, :944, and the c0 cache at :763) | Unfilled sd imbalance in [BOS_n, CTS_n]? | check_to = cts_idx = CTS_ESTABLISHED.idx (:521), the anchor. The moment `meta["confirmed_at"]` (market_structure.py:1493) is not used | anchor | Yes, when start == anchor. It is truly unknowable only at lag 0 (31 of 34 EST rows). 1 measured flip: conf sub2 cyc0 EST 2368 (lag 0), activation 2368→2369 via row 18 [G2; INBOX:36-38] | fix, only via `knowable_at = confirmed_at` (disputed) | Keep the fill as-of at the anchor until Plan E E3a. Never key the cut on check_to: lag-1 cycles (1223/1224, 2828/2829) would drop instances that are knowable. Test fixtures lack confirmed_at (test_cross_cycle_fib.py:40-46; test_main_versioned_cross.py); use direct indexing, not `.get`. H1 cycle≥1 activation happens once, at EST (:1242-1251, :1283-1288), so an exclusion there deletes the fib. See Q3. |
| 17a | fib_tracker.py:1318, c0 re-snapshot, as read immediately at :1330 and :1351 (Scenario-1 activation at the same update) | Activate cycle 0 now? | [BOS_0, update idx]; check_to = CTS_UPDATED.idx (:1131) | mixed (raw path: moment) | Yes, when start == the update candle (H1 706 and 709 in scope, 0 flips) | fix | `knowable_at = ev.idx`. It must not also poison 17b (see Q2). |
| 17b | The cached `c0["has_unfilled"]`, written at :763 and :1318 and read as cond2 at cross_cycle_fib.py:153-154 (via fib_tracker.py:182) | cond2 at the CTS_1 EST decision | Fill horizon CTS_0 | mixed | No at the decision candle: CTS_1 EST >= CTS_0+1 | already_bounded (disputed) | Keep this value unfiltered. If 17a is fixed, either store two values or recompute at use. Q2. It pairs with row 12 (LANDMINES.md:188-223). |
| 18 | fib_tracker.py:1183, cross_cycle cycle-0 first activation on update | Unfilled sd in [BOS_0, update candle]? | check_to = CTS_UPDATED.idx | mixed (raw path: moment) | Yes. Measured flip: sub3 raw 2650, whose only instance is (2650, 2651); activation 2650→2651 and meta activated_at 61→62 in both lenses [G2; INBOX:38] | fix | `knowable_at = ev.idx`. The worked example at FIB_LIFECYCLE_SPEC.md:919-925 goes stale. |
| 19 | fib_tracker.py:1462, `_update_fib_cts` normal branch (callers :1169, :1251, :1288, :1327, :1590, :1594, :2525) | Is the fib's own range still unfilled after the CTS extends (reactivate / deactivate)? | [bos, cts_idx]; check_to = CTS_UPDATED.idx | mixed (raw: moment; new pattern-path rows sit at apply; only REGRESS rows are anchor-timed, 0 on this window) | Yes: 41 of 246 calls in scope, 0 flips [G3] | fix | `knowable_at = ev.idx`. Re-check gap: the lock at CTS_CONFIRMED reads a stale `active` flag (:1667, :1681, :1747, :1754). See Q3. |
| 20 | fib_tracker.py:1535, `_update_cycle1_main` cond2 (cycle-1 own imbalance) | Is the cycle-1 range unfilled as of the update? | [cycle1_bos_idx, cts_idx] | mixed (all 6 H1 calls are moment-timed) | Yes: (751,751) at 751 and (760,761) at 760; 0 flips | fix | Change in lock-step with row 21. Same re-check-gap caveat. |
| 21 | fib_tracker.py:1571, create-on-fail `normal_has_unfilled` | Same question as row 20 | Same as row 20 | mixed | Reached only when cond2 is False | equivalent | Lock-step with :1535. Unreachable on H1 (CROSS_CYCLE_FIB_SPEC.md:370-373). |
| 22 | fib_tracker.py:1527, `_update_cycle1_main` cond1 (cycle 0 @CTS_0) | Recompute cycle-0 liveness | [BOS_0, CTS_0]; check_to = c0_cts_idx, a past fill horizon | final_state_retrospective | No, measured against the knowledge candle (update > the CTS_0 lock; 0 of 6 in scope) | already_bounded (disputed: G3 said equivalent) | Must never get a filter keyed on check_to. Under a consistent rule it is always True whenever it runs. |
| 23 | fib_tracker.py:1540, cond3 @BOS_1 | Did BOS_1 fill cycle 0? | [BOS_0, CTS_0] as of cycle1_bos_idx | anchor | No: BOS_1 > CTS_0 | already_bounded | None. |
| 24 | fib_tracker.py:1436, :1445, :1451, the `_update_fib_cts` cross branch (:1421-1457) | (dead code) | n/a | n/a | Dead. `is_cross_cycle` reads `meta["cross_cycle"]` (:1421), which is only written at :935 (stored under a 4-tuple key, cross_version=0 at :940) and at :2310. Every caller passes a 2-tuple key | out_of_scope | Delete in a separate hygiene change. Strike ":1446" from PLAN_E_inputs.md:633. |
| 25 | fib_tracker.py:1056-1099 `_activate_fib`; :2317-2329 `_m15_create_cross`; :2023-2055 `_obsolete_prev_cycle_all_fibs` | Lifecycle writes on the activating cycle and the previous cycle | activated_at = cts_idx (anchor) on most paths; current_candle on the THRESHOLD path | mixed | Inherits the caller's verdict | defer_to_plan_e (activated_at used as a time is an E3 item) | No change. The predictions must list the previous cycle's end_idx / end_reason flips. |
| 26 | cross_cycle_fib.py:117 via `on_cts_threshold_updated` (fib_tracker.py:2107, then :2158) | Does the pre-established target cycle have its own unfilled imbalance? | [prospective BOS, current_candle]; check_to = CTS_THRESHOLD_UPDATED.idx (:2090) | processing_candle (emitter market_structure.py:1767-1790) | Yes: 8 of 46 calls in scope. 1 flip: counter sub5 THRESHOLD 3806, whose only instance is (3806,3806) [G3, G4] | fix | `knowable_at = current_candle`. Measured effect [INBOX:38-40; G4]: counter fib_lifecycle goes from 9 to 8 rows (the sub5 cyc1 cross is never created; the sub5 cyc0 end moves from 3806/new_cycle to 4083/same_dir_replacement), and counter POIs from 13 to 12 (IC 3654, which is the parked fallback POI). No later event re-checks it. See Q3 and Q4. |
| 27 | cross_cycle_fib.py:117 via raw CTS_UPDATED (fib_tracker.py:1218; H1 §11b path :1281 then :1966) | Same question, established target | check_to = CTS_UPDATED.idx | processing_candle | Yes: 8 of 63 calls in scope, 0 flips | fix | `knowable_at = ev.idx`. |
| 28 | cross_cycle_fib.py:117 via pattern-path CTS_UPDATED (same callers) | Same question | check_to = the pattern's extreme candle | anchor | 1 of 5 calls in scope ((455,456) at 455, where anchor == apply); 0 flips | defer_to_plan_e | No moment field is recorded (ARCHITECTURE.md:100). The G3 verifier notes that new pattern rows have idx == apply and that duplicate rows repeat the raw evaluation, so only REGRESS rows lag. I did not verify this. |
| 29 | cross_cycle_fib.py:117 via the EST callers: fib_tracker.py:609 (sub cycle≥1), :880 (H1 cycle-1 Scenario-2 cond1), :1991 and :2009 (§11b peek and apply) | Own imbalance at EST | check_to = CTS_ESTABLISHED.idx (anchor) | anchor | Yes when start == anchor. 2 of 14 sub EST calls in scope: (193,193) at lag 0, which is unknowable; sub3 cyc1 (2828,2828) at lag 1, which is knowable at 2829. 0 flips | fix, only via `knowable_at = confirmed_at` (disputed) | Same inputs as row 16, so it must use the same knowable_at. Otherwise the cross and single decisions diverge (:911-957; :719). Pass the keyword as an optional argument, default None. |
| 30 | cross_cycle_fib.py:156 (current-policy dead-cycle walk) and :139-141, :160-161 (`dead_cycles` latch) | Is prior cycle k still live? | [BOS_k, CTS_k] as of current_candle | mixed | No. cts_by_cycle is filled only at CTS_k CONFIRMED (fib_tracker.py:1640-1642), so current > conf_k > CTS_k. 0 of 162 calls in scope | already_bounded | No filter. Never key on check_to at the EST callers. Under R2 the latch would mark a live cycle dead permanently. |
| 31 | patterns/imbalance.py:159-166 `has_unfilled_imbalance` and common/types.py:216-224, the `is_filled` empty scan | The shared chokepoint | check_to is both the fill as-of and, implicitly, the existence as-of | mixed | Yes, for every caller whose window ends at check_to | defer_to_plan_e (applies to a filter keyed on check_to by default) | A default keyed on check_to stays wrong even after Plan E, because of the retrospective as-ofs at rows 22, 23 and 13. Add a keyword-only knowable-at parameter (default None; R1: count an instance only if `start_idx < knowable_at`). Keep `is_filled` unchanged (test_imbalance.py:259-268). |
| 32 | imbalance.py:169 `get_unfilled_imbalances`; imbalance.py:118 `has_imbalance_in_range` | n/a | No production caller (the import at fib_tracker.py:25 is unused) | not_applicable | n/a | out_of_scope | Document that neither has an existence as-of, or delete them in a separate change. |
| 33 | zones/zone_proximity.py:487, `_try_trigger_at_candle` via `poi_active_as_of` | Is the POI active at scan candle i? | activation_history as of i (H1 main only) | processing_candle | Inherited: at 997 the POI was active on an unknowable instance, but no trigger fired | equivalent | None. Triggers are identical under R1 and R2 [G6]. A delta requires a base sd:POI trigger on a candle whose activity is removed. After such a divergence the trigger set is path-dependent (Rule-3 latches at :390-391, :422-425, :450-454). |
| 34 | zones/poi_lifecycle.py:23-92 | Pure history walks | Caller's idx | mixed | Inherited | equivalent | None. |
| 35 | H1 trigger chain: orchestrator.py:344-364 (WVMI gate); multitf/uc1_trigger.py:66-121; subsequent_confluence_trigger.py / subsequent_counter_trigger.py :104-150; orchestrator.py:826-900 plus lifecycle_sweep.py; orchestrator.py:957-968 | Downstream followers of trigger idx | Processing candles | processing_candle | Inherited via row 33 | equivalent | Unchanged under R1 and R2 on this window [G6]. |
| 36 | multitf/entity_df_mutation.py:253-266 (POI mirror) and :268-294 (fib mirror) | idx translation | n/a | not_applicable | Inherited | equivalent | None. |
| 37 | Charts: charting/_zone_render.py:108-125; export_plotly.py:2281-2505, :2094-2110, :2509-2682; export_m15_chart.py:1540-1690, :2457-2600, :2674-2687 | Render engine moments | activation_history and final fib state | final_state_retrospective | Inherited | equivalent | Predicted: stretch starts and confirm lines shift 953→954 and 997→998 (H1), 4118→4119 (M15); the 1020 hover zone_confirmed_idx goes 997→998. Counts only change if a stretch shrinks to one candle, because the fill is dropped at `sx1 <= sx0`. |
| 38 | Imbalance highlight: export_plotly.py:272-345, export_m15_chart.py:890-925; also export_m15_chart.py:1098 (unused) and style_registry.py:130-185 | Draw the c2 location | none | not_applicable | n/a (a pattern location, not a time) | out_of_scope | Rendering rule. Add one cross-link line in CHARTING_SPEC.md:89-94. |
| 39 | Exports and logs: debug/export_imbalances.py:18-30, export_zones.py:7-41, export_fib_lifecycle.py:43-62, run_replay.py:215-219 (`_final.csv`), orchestrator.py:429-448, debug/zone_proximity_diag.py:118 | n/a | n/a | not_applicable | n/a | write_only | These pin byte-identity for /compare: `is_imbalance` (218 H1 rows) and `_imbalance_instances.csv` (167 rows). Stash no new `inst.meta` key unless the prediction covers it. |

## (1) Scope options

**Option A: POI sweep only (row 1).** This is the only code change. Rows 5 and 33-37 follow without code changes.
- Measured (prior session, confirmed by [INBOX:33-34] and G1): 21 of 24 CSVs are identical, and exactly the 4 POI rows change.
- It does not depend on Plan E. The as-of is a processing candle, and first_active has been a moment since Plan D.
- It breaks a documented identity. ARCHITECTURE.md:212-222 and poi_zones.py:516-518 define the sweep as `has_unfilled_imbalance(df, ic_idx+1, t, check_to_idx=t, direction=sd)`. The fix is to add the default-None knowable-at keyword (byte-identical for all 16 callers) in the same commit and restate the identity with `knowable_at=t`, or else rewrite it as the sweep's own predicate.

**Option B: fib-side sites with a real moment, via the opt-in keyword.**
- Sites: rows 16, 17a, 18, 19, 20, 21 (lock-step), 26, 27 and 29. Not rows 22, 23, 28 or 30.
- Measured effect of the "downstream fib only" variant [INBOX:36-40] (flips at rows 16, 18, 26):
  - conf fib_lifecycle: 2 rows change (sub2 cyc0 activated_at 53→54 and gains `activated_on='update'`; sub3 cyc0 61→62).
  - counter fib_lifecycle: 9→8 rows (sub3 activated_at 61→62; the sub5 cyc1 cross is gone; the sub5 cyc0 end changes).
  - counter POIs: 13→12.
  - H1 is unchanged.
- The deltas can be separated (the combined variant is the exact union [INBOX:42-43]). So B1 (rows 16-21: meta cells only) and B2 (rows 26-29: a fib row and a POI row removed) can be two /compares.
- Anchor as-of sites: rows 16 and 29 have an anchor as-of. They are exact only with `knowable_at = confirmed_at`; the fill as-of stays on the anchor until Plan E E3a. Row 28 (pattern-path UPDATED) stays deferred.

**Option C: MS in-flight (rows 10-12). Recommendation: no change.** A filter would only move replay toward "a live engine running this code on its refresh schedule" and away from the per-candle rule in the spec (Q1). Measured 24/24 identical [INBOX:35].

**Option D: downstream IC identification (row 7).** This belongs to Plan E §2.4 item 8. It is 24/24 identical even if filtered [INBOX:41-42].

**Not recommended: a central filter in `has_unfilled_imbalance` keyed on check_to.**
- It would reach all 16 call sites.
- It would be wrong at the anchor sites (7, 16, 29) at lag ≥1, at the retrospective as-ofs (13, 22, 23), and on the in-flight path (10-12).
- The wrapped-R1 suite passing (739 + 1 xfail) shows missing test coverage, not correctness.

**Recommendation:** land A now, with the keyword and the doc rule. Decide B after Q2-Q4. No change for C. D goes into Plan E.

**Decisions needed:**
- **Q1. The reference model for cached snapshots.** Option (a) is the per-candle rule in the spec (MARKET_STRUCTURE_SPEC.md:203-212, "deliberate approximation"): no filter. I recommend (a). Option (b) is "same code, live refresh schedule": thread the refresh moment through `PoiInnersResolver` (market_structure.py:62-67).
- **Q2. What cond2 means.** Option (a): the fill horizon is CTS_0, with knowledge judged when the value is used. This keeps "cond3 implies cond2" and CROSS_CYCLE_FIB_SPEC.md:165-167 true. I recommend (a). Option (b): knowledge as of CTS_0 itself, which needs a two-value cache at rows 17a/17b and a matching change at row 12.
- **Q3. The event-driven re-check gap.** FibTracker only re-evaluates at CTS events. The lock reads a stale `active` flag, and H1 cycle≥1 activation happens once. So a c3 exclusion can delete a fib, cross or POI rather than delay it by one candle. That matches today's event granularity (G4 verifier). Adding a c3 re-evaluation trigger would be a separate design change.
- **Q4. The parked fallback POI.** B2 removes the parked fallback-POI instance (counter sub5 cyc1 IC 3654) on this window. Should B2 land before that item is picked up?
- **Q5. Naming.** The accessor or keyword needs a moment name per the GLOSSARY Naming Standard (candidates: `formed_at`, `established_idx`). Avoid the pool's existing `knowable_at_idx`.

## (2) R1 vs R2

R1 is recommended: an instance is visible as of K iff `start_idx < K`. The exact form overlaps on [start_idx, min(end_idx, K−1)] [INBOX:31]. Overlapping on the full span gives the same result for every current call shape (G7).

Why R1 and not R2:
- **Live prefix.** A live engine sees the prefix from start+1. In the sweep, R2 gives 997→1000 and 4118→4120 where R1 gives 998 and 4119.
- **The boolean is exact.** The prefix's only scanned candle is its own c3, which cannot arm (types.py:228-235 with fill_threshold > 0).
- **R2 flips things wrongly:** H1 EST 703 (instance 702-703), M15 sub4 EST 3306, 4 calls at row 19, and cross_cycle_fib.py:117 at 1288.
- **Tests.** A wrapper-level R2 fails 10 tests in test_poi_activation_moment.py (instance (8,9) at anchor 9).
- **Dead-cycle latch.** In the walk, R2 would latch a live cycle dead permanently (row 30).

Caveats:
- **Degenerate gaps.** `is_filled` returns True when gap_size <= 0 (types.py:196-197). A prefix gap and its merged gap can differ in degeneracy, in either direction (G7 verifier's counterexample). There are 0 cases on this data (H1: 0 of 167; M15: 0 degenerate prefixes [INBOX:44]). The sweep's `gap_size > 0` filter at :915 excludes a degenerate merged run even if its prefix is valid.
- **Final merged bounds too early.** R1 enters an instance with its final merged bounds before the run's tail is known. The G6 verifier checked 0 within-run partial fills on H1's 43 merged runs; M15 is unchecked.
- **The rule depends on fill_threshold > 0** (orchestrator.py:307).

## (3) Docs to change and tests affected

**Docs (same commit):**
- **IMBALANCE_FILL_SEMANTICS.md** (canonical home):
  - a new "Knowability (c3 rule)" section before :11, carrying the degenerate-gap caveat;
  - :34-36 on monotonicity: under c3, "has unfilled" is not monotone in the as-of;
  - :43 split the row;
  - :65-81 the keyword and the full caller list;
  - :83-87 mark as dead;
  - :97-102 add the sweep row and the in-flight path;
  - :104-116 regenerate (stale lines; add fib_tracker.py:1183 and cross_cycle_fib.py:117/149/156; an as-of-kind column; note the cond label swap);
  - :118-128 point to :2050 (`_capture_cycle0_snapshot` does not exist);
  - :157-182 the enter side;
  - :186-203 a dated subsection with the deltas, and a fix to :199-203, where "live candle" is wrong for EST/UPDATED.
- **Code docstrings and comments:** imbalance.py:143-157 ("permissive" is stale); types.py:194; poi_zones.py:226-228, :516-518, :809-812, :817-819, :924-925.
- **POI_ZONES_SPEC.md:** :49-61, :80-85, :100-101 (the scan starts at the last c3), :263 (undefined `imbalance_idx`), :272-273, :289, :427-430.
- **ARCHITECTURE.md:** :166-167 (define "form"), :212-222 (the identity), :136-142 (separate the raw path from the anchor path); optionally a THRESHOLD row under :106.
- **GLOSSARY.md:** :145-153 (add a knowable-at row; :153 is the stale single-stroke fill definition).
- **GOTCHAS.md:840-862:** the existence case goes away; merged bounds are harmless except for degenerate gaps.
- **LANDMINES.md:** :1512-1517 and :1535-1538 (site list); :2098 (stale name); :2112-2119 (residual wording); :188-223 (note the in-flight caller exception if the keyword is added).
- **MARKET_STRUCTURE_SPEC.md:** :203-212 (the consumer-gate bound from row 9); :326-332.
- **CROSS_CYCLE_FIB_SPEC.md:** :90-95 and :133-135 (current_candle is a moment only on THRESHOLD and raw UPDATED); :162-170 only if Q2 = (b).
- **FIB_LIFECYCLE_SPEC.md:** :139, :337-339, :364-372; :919-925 if B lands; stale references at :97 and :582-583.
- **CHARTING_SPEC.md:89-94:** add the cross-link.
- **Plans:** PLAN_E_inputs.md :86 (mark done), :633, :638, :662-665 (annotate items 3-6 and 8-9); PLAN_D_poi_activation_moment.md:257 (DONE marker).
- **Memory:** project_zones_timing_audit_20260922.md:87-95; MEMORY.md priority 1(a); project_item_3_poi_lifecycle.md:39 and :110 (the identity); project_sub_structure_pool_architecture.md:149-153. Drain _INBOX.md.

**Tests:**
- No existing test pins either leak. The suite passes unchanged under sweep-R1, sweep-R2 and wrap-R1 [G1, G6, G7].
- New tests to write first:
  - A direct sweep test where c2 == first_active, plus a merged run that counts from start+1 (this separates R1 from R2). Add near test_poi_activation_moment.py:255.
  - The full history of `_make_multicycle_data` POI (0,2) IC 12: today [(15,A)]; under the fix [(15,A),(19,D,imbalance_filled)].
  - If the keyword is added, after test_imbalance.py:345: knowable_at == start gives False; start+1 gives True; merged (1,3) at 2 gives True; a degenerate-gap case.
  - For B: a lag-0 EST with c2 == anchor; a lag-1 EST (anchor 9, moment 10) that still activates; a raw update at :1183 with c2 == t; a :1462 deactivate/reactivate case; a positive pre-established THRESHOLD cross (no coverage exists today).
- Relabel test_poi_lifecycle.py:17-24 ("as of save 0a4eadc, pre-c3") and the comment at test_poi_activation_moment.py:251.
- For B, the fixtures need confirmed_at: test_cross_cycle_fib.py:40-46 and test_main_versioned_cross.py.
- test_poi_activation_moment.py (10 tests) acts as the R2 tripwire.

## (4) Latent bugs and surprises (not to be fixed now)

- **Reconciled:** the session's first "ENGINE LOOK-AHEAD" framing of the MS in-flight path was corrected to row 10's
  ruling (compute-time, not decision-time; §2.3) — in `_INBOX.md` and here.
- **Dead code:**
  - the fib_tracker.py:1421-1457 cross branch;
  - the `.get` defaults at :1433, :1442, :1524 and :1532;
  - `get_unfilled_imbalances` and `has_imbalance_in_range`;
  - the unused import at fib_tracker.py:25.
- **The cond1/cond2 labels are swapped** between the code (:1430-1448, :1522-1538) and the docstring (:126-129) and POI_ZONES_SPEC.md:175-177.
- **The MS and FibTracker c0 snapshots can diverge.** `_update_cycle0_data` overwrites unconditionally (market_structure.py:2054), while FibTracker only updates when cts_idx increases (fib_tracker.py:1310). They split on a pattern-path CTS_UPDATED that moves backwards (REGRESS), which is also the only anchor-timed pattern row that gets past the fib guard.
- **Chart and proximity oddities:**
  - The zone_proximity Rule-3 latches make downstream deltas path-dependent.
  - `poi_active_as_of` evaluates at end of candle, while the narrow gap uses start of candle (GOTCHAS:902).
  - The chart hover picks a POI by nearest inner price, so it can pick an inactive twin (export_plotly.py:2094-2110, export_m15_chart.py:2674-2687).
  - Single-candle stretches are dropped (`sx1 <= sx0`).
- **Misleading or stale wording:**
  - The `_build_geometry` docstring at entity_df_mutation.py:1053 implies run_cap is the sub's end; it is the data edge.
  - LANDMINES.md:2113-2115 is loosely worded.
  - `*_THRESHOLD_UPDATED` is missing from the ARCHITECTURE ev.idx table; G4 verified that its idx is the processing candle.
- **Other:**
  - `inst.meta` is mutated on instances shared across projections (types.py:156-166), so the debug export shows the last writer.
  - `ImbalanceInstance.start_idx` is the first c2, not the pattern's first candle. That is a Plan E naming input.
  - The bearish highlight colour in CHARTING_SPEC differs from the code (export_plotly.py:325).

## (5) Disagreements resolved

1. **MS in-flight (rows 9-11).** The positions were:
   - already_bounded: G1, the G3 verifier, G4 (reader and verifier), the critic verifier;
   - fix or defer: the G3 reader;
   - defer: the G5 reader and the critic reader;
   - fix: the G5 verifier.

   **Ruling: already_bounded.** I re-read market_structure.py:1975-1981: the gate is `i > st.cts.idx` and `cts_cycle_id > 0`, and :1957 is the only reader. The refresh uses `int(st.cts.idx)` (:2007). The windows at poi_zones.py:211 and :229-233 end at cts_idx. The spec calls the snapshot an approximation of a per-candle check (MARKET_STRUCTURE_SPEC.md:203-212), and against that reference the instance is knowable at every candle where it is used. The G5 verifier's "same code, same refresh schedule" reading is legitimate, so it is kept open as Q1. It makes 0 difference on this window [INBOX:35].

2. **market_structure.py:2050 (row 12).** Positions: G1 already_bounded, G5 reader defer, G5 verifier fix, G4 out_of_scope. **Ruling: already_bounded at the decision candle.** It is consumed only through fib_tracker.py:182 into cross_cycle_fib.py:153-154. The semantics go to Q2, together with rows 17a/17b.

3. **The EST callers (rows 16 and 29).** Positions: the G4 reader and G3 said defer; G2 and the G4 verifier said fix. **Ruling: fix, but only via confirmed_at.** It is emitted at market_structure.py:1493 and the event is in scope at fib_tracker.py:521. Only the fill as-of depends on Plan E.

4. **fib_tracker.py:1318 (row 17).** Positions: the G2 reader said fix; the G2 verifier refined it; G4 said out_of_scope. **Ruling:** split into 17a (fix) and 17b (already_bounded). There are two consumers, :1330/:1351 and cross_cycle_fib.py:153-154.

5. **fib_tracker.py:1527 (row 22).** Positions: G2 already_bounded, G3 equivalent. **Ruling: already_bounded,** measured against the knowledge candle. G3's point that it is always True is kept as a note.

6. **fib_tracker.py:1436, :1445, :1451 (row 24).** **Ruling: dead.** I verified that the `"cross_cycle": True` writes are only at :935 (under the 4-tuple key from :940) and at :2310.

7. **`_compute_fill_idx_cache` (row 4).** G6 said write_only; G1 and the G6 verifier said already_bounded. **Ruling: already_bounded,** because the tuple drives poi_zones.py:923 and :928-933.

8. **zone_proximity.py:487 (row 33).** The G6 reader claimed the effect is monotone. I re-read the latches at zone_proximity.py:390-391, :422-425 and :450-454: after a divergence, a trigger can appear on a candle where the base run had none. The classification stays "equivalent", and the prediction criterion is the precondition in row 33.

9. **poi_zones.py:448 and :198 (row 8).** I adopted the verifier's corrections: a fib can be prevented from being created, :198 is looser than :448, and :198 has a second caller in-flight (:1098).

10. **entity_df_mutation.py:1060-1067 (row 15).** G6 said run_cap is the sub's natural end. **Ruling: it is the data edge** (:1138, per the G5 reader and G6 verifier).

11. **data_bridge.py:82.** The critic's claim that this is reachable was refuted by its verifier: Phase 2 runs sid 0 on the intra path (fib_tracker.py:155), and the snapshot is read only when i > cts. Out of scope.

12. **The critic's MS doc row.** The critic said defer; the verifier said already_bounded. **Ruling: record the row-9 bound in MARKET_STRUCTURE_SPEC.md:203-212 in this commit.** Do not add the proposed LANDMINES line "measured inert on this window": it is a provable bound, not an observation.

13. **Tests.** Four test records were relabelled from already_bounded to equivalent, because each does reach an unknowable instance: test_imbalance.py:259-268, test_poi_activation_moment.py:151-162, test_ms_bounded_equals_truncated.py and test_render_sub_projection.py:724-801 (G7 verifier).

14. **G7's claim that R1 is exact "whatever the bounds".** Corrected with the degenerate-gap caveat (types.py:196-197; the G7 verifier's counterexample).

15. **PLAN_E_inputs.md:633** says current_candle is a real moment only on the THRESHOLD path. This is overruled: the raw-path CTS_UPDATED idx is a moment too (ARCHITECTURE.md:100).