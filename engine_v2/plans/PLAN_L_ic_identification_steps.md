# Plan L — IC identification in steps: the restructure + the ranking rules (Week 9 Part 1 item 1)

**Status: L1 COMMITTED `b5c3613` + SAVED S_L1 `20261010_114846_b5c3613` (2026-10-10, §9); L2 COMMITTED `6d8cc67` + SAVED S_L2 `20261010_155816_6d8cc67`
(2026-10-10, §10 — `/compare` vs S_L1 == §4.3; landing-reviewed 506,850 tok, all folded; suite 1609, mutants 56 / 56; the
chart review approved). PLAN_L COMPLETE — the IC identification revamp done but Step 4 (deferred, register §D.2).** §6 names APPROVED, §4.1 runner-ups settled (PLAN_H §6
ICR-13 / ICR-14); the cold review RAN (§8: 0 BLOCKER, 4 MAJOR, 17 MINOR — all folded below). The IC
identification revamp: the user's write-up verbatim `plan_h_inputs/ic_identification_revamp_spec.md`; every decision
PLAN_H §6 ICR-S, ICR-1 … ICR-15 (2026-10-09 → 10); the reading, the shadow measurement and the TRIAL replay
PLAN_H §8.9 "The IC identification revamp". Baselines: L1 vs **S_TBU** `20261009_171255_5b5c307` (code `92b8b61`);
L2 vs **S_L1** `20261010_114846_b5c3613` (code `b5c3613`); after L2: **S_L2 `20261010_155816_6d8cc67`** (= `LATEST_<branch>` from 2026-10-10). Register `IDEA_PARKING_LOT.md` §D.1 item 1; Step 4 → §D.2 (deferred).

Planning shorthand: L1 / L2 = the two landings; ICR-* = PLAN_H §6 rows (each has a descriptive title there).

---

## 1. The rule as decided

Per POI record (a gated fib record, BOS anchor `b`, CTS anchor `c`, direction `sd`), in order — each step, and each
rule set inside it, runs sequentially (ICR-S):

- **Step 1 — the IC candidates.** Every candle of the fib's `b … c` window (the scope, not a condition) whose AGREED
  direction is `−sd` (condition 1) is checked against the other three conditions, all applied TOGETHER (PLAN_H RX1-F):
  (2) the IC reference condition (POI_ZONES_SPEC §3.2 — price side + before-the-break side), (3) an unfilled sd
  imbalance after it through `c`, (4) it touches the 61.8–80 zone. An **IC candidate** passes all four. None →
  **no IC**, reason `no_ic_candidate` (Step 4 would run — deferred).
- **Step 2 — the institutional pattern** over the IC candidates (POI_ZONES_SPEC §3.4, unchanged): the best
  (rank, shape); if two or more patterns tie, the **tiebreak rules** over the tied ICs: (1) the immediate unfilled
  imbalance (a filter), (2) the IC's candle type maru > normal > pinbar, (3) the most recent before the imbalance;
  the first rule to leave one decides. Any pattern → the IC, `ip_outcome = pattern`. None → `ip_outcome =
  no_pattern` → Step 3.
- **Step 3 — the ranking rules** (NEW) over the UNFILTERED Step 1 list: (1) exclude pinbars — none left → **no IC**,
  reason `all_ic_candidates_pinbar` (Step 4 would run — deferred); (2) the immediate unfilled imbalance (a filter:
  one passes → it; two or more → only they go on; none → all go on); (3) the candle type maru > normal; (4) the most
  recent candidate in play before the earliest unfilled sd imbalance after the OLDEST candidate STILL IN PLAY after
  rules 1–3 (ICR-8; a candidate may be the gap's c1 — two neighbours sharing one gap → the one closer to it, ICR-1).
  The first rule to leave one decides (`ranking_rule`); Step 1 found exactly ONE IC candidate and it is not a pinbar →
  `ranking_rule` empty (ICR-10; a single PINBAR candidate → no IC, `all_ic_candidates_pinbar`). Rule 4 always decides when reached (the oldest in play has an unfilled imbalance after it — condition 3).
- **Step 4 — the lower-TF fallback: DEFERRED** (register §D.2, maybe never): no IC → no POI.

**Shared rule helpers, separate rule sets (ICR-9).** A rule that is the same in both sets is ONE function with ONE
parameter set, given a different input list: the immediate unfilled imbalance (tiebreak rule 1 = ranking rule 2; ONE
window parameter `immediate_imbalance_candles` = 2) and the most recent before the imbalance (tiebreak rule 3 = ranking
rule 4). The candle type is ONE function with an optional pinbar (tiebreak rule 2 with it, ranking rule 3 without).
Tiebreak rule 1 receives the ICs of the patterns tied on the best (rank, shape); ranking rule 2 receives what ranking
rule 1 left of all Step 1 candidates. Nothing one rule set computes feeds the other.

**Consumers.** ONE selector for the drawn POI and MS's in-flight read (PLAN_H D2; ICR-12). A Step-3 POI's activation
floor has no pattern term (ICR-2): it goes live from max(the cycle's CTS-established moment, the IC, the structure's
start) under the unchanged live conditions (a formed unfilled sd gap after the IC, the touch of the as-of zone).
Chart: hover + CSV only (ICR-7).

## 2. Shape — two landings, one cause per `/compare`

| Landing | What | Selection results | `/compare` vs | Predicted |
|---|---|---|---|---|
| **L1 — the restructure + the names** | Steps 1–2 as the user's steps (§3); the names (§6); the debug CSV lists every opposite-direction candle with one outcome column per condition (ICR-4, ICR-11) | IDENTICAL (Step 3 not wired) | S_TBU | every CSV byte-equal except the 3 IC-candidate CSVs (renamed; the old rows equal on the shared columns + 54 new rows + 2 new columns); charts byte-identical (§3.3) |
| **L2 — the ranking rules** | Step 3 in both consumers; `ip_outcome`, the `ranking_*` fields, `all_ic_candidates_pinbar`, "IC: none", the hovers (§4) | 7 records gain an IC | L1's save | = the trial replay (§4.3) + the format |

Why split: L1's proof is byte-identity — inside L2's expected moves a slip in the restructure would hide (memory
`feedback_one_cause_per_compare.md`). Each landing: tests → mutants (`mutant_runner.py`, full suite each) →
`/compare` == the prediction → landing review → `/commit-save` (local, no push).

## 3. Landing 1 — the restructure + the names (selection results identical)

### 3.1 The code shape (`zones/poi_zones.py`)

- **`check_ic_conditions(df, fib_state, config, *, ic_reference) -> Tuple[ICCandleCheck, ...]`** — Step 1's
  per-candle work: every candle of `b … c` with agreed direction `−sd` (index order; today's `idx not in df.index`
  skip kept), each an **`ICCandleCheck`** = `ic_idx`, `has_unfilled_imbalance_after` (condition 3 — today's test,
  `has_unfilled_imbalance(idx+1 … c, check_to_idx=c, sd, evaluated_at=None)`), `ic_ref_reason` (condition 2 —
  `ic_reference_reason`, unchanged), `touches_zone` (condition 4 — `fib_zone_bounds` + `ic_touches_zone`),
  `immediate_imbalance_first_c2_idx` (the per-candle fact, set on every CLASSIFIED check — below), `pattern:
  Optional[IPResult]` (None = not classified), and the property `is_ic_candidate` = conditions 2–4 pass (condition 1
  is the row set) — it replaces the 2-argument predicate `is_ic_candidate(touches_zone, ic_ref_reason)`.
  Invariants kept from `select_ip_ic` (B-m10): the frame's stamp / candle params / leg cap are read BEFORE the
  active-or-locked guard (the fail-loud pins); not active and not locked → no checks.
- **`find_ic_candidates(checks) -> List[int]`** — Step 1's answer: the IC candidates (`is_ic_candidate`), index order.
  (Today's `find_ic_candidates` — direction + condition 3 only, candle direction by default — is the unwired
  V30 / V60 / V90 path's finder: renamed **`find_ic_variant_candidates`**, kept with `select_ic_variants` and their 3
  test sites, ICR-5.)
- **`select_institutional_pattern(df, fib_state, config, ic_candidates) -> (pick, tiebreak)`** — Step 2 (the IC
  candidates' checks carry their patterns and immediate gaps — nothing re-derived, A9):
  the best (rank, shape) among the IC candidates' patterns, then **`apply_tiebreak_rules(tied, …)`** — today's chain,
  built on the shared helpers below; outputs unchanged (`TIEBREAK_*` names, the 4-tuple).
- **The shared rule helpers (ICR-9)**: `filter_immediate_unfilled_imbalance(in_play, immediate_c2) -> List`,
  `prefer_candle_type(in_play, df, *, include_pinbar) -> List`, `most_recent_before_imbalance(in_play, df,
  fib_state, config) -> (pick, reference_first_c2)`. Per-candle fact, measured once per checked candle:
  `immediate_unfilled_imbalance_first_c2_idx` (unchanged; its window = `config.immediate_imbalance_candles`, renamed
  from `ip_immediate_imbalance_candles`). The structural claim "with ICs that all passed rule 1 the last rule picks the
  oldest" (TBU-4) is Step-2-only → it moves from the helper's docstring to `apply_tiebreak_rules`'.
- **`select_ic(df, fib_state, config, *, ic_reference, classify_all=False) -> ICSelection`** — runs Step 1 → Step 2
  (→ Step 3 in L2). `ICSelection(checks, pick, tiebreak)` + the view `ic_candidates` (the checks with
  `is_ic_candidate`) — as BUILT in L1; L2 adds `ip_outcome`, `ranking`, `no_ic_reason`. **What `checks` holds (B-M1 — a CONTENT change, not a
  rename):** EVERY agreed-opposite candle of the window, with `classify_all=False` too (today `candidates` with
  `classify_misses=False` holds only the classified IC candidates); `pattern` + `immediate_imbalance_first_c2_idx`
  are set on the IC candidates (`classify_all=False` — MS's read: the imbalance scans stay as today, B-m10) or on
  every check (`classify_all=True` — the drawn record + the debug CSV, ICR-11; today `classify_misses` classifies the
  base set only). The pick reads only `ic_candidates` — a non-candidate's pattern is never picked (D2 pins with
  teeth, §3.4). Renamed with it: `IPSelection.candidates` → `ICSelection.checks`, `IPCandidate.ip` →
  `ICCandleCheck.pattern`, the kwarg `classify_misses` → `classify_all`. The early return (no fib / not active and
  not locked) → (L2) `no_ic_reason = "no_ic_candidate"` (A6 — today's derived reason for such a record).
  As built, `check_ic_conditions` also takes the frame's agreed direction (`agreed=`, computed once by `select_ic`).
- **Callers**: `select_ic_records` (was `select_ip_records`), `derive_poi_zones`, `derive_ic_candidates` (was
  `derive_ip_candidates`), `poi_inner_for_fib` (MS) — each calls `select_ic`; the record gate, the CTS_ESTABLISHED
  lookup, the final-reference call and the fail-loud scope unchanged.

### 3.2 Audit — every site (✓ = edited in L1)

| Site | Today | L1 |
|---|---|---|
| `zones/poi_zones.py` | `find_ic_candidates`, `is_ic_candidate`, `IPCandidate`, `IPSelection`, `select_ip_ic`, `IPRecordSelection`, `select_ip_records`, `IPCandidateRecord`, `derive_ip_candidates`, `IPSelectionError`, `POIConfig.ip_immediate_imbalance_candles` | ✓ §3.1 + §6 names; `IPCandidateRecord` → `ICCandidateRecord` + `has_unfilled_imbalance_after`, `is_ic_candidate` fields |
| `patterns/institutional_pattern.py` | docstrings name `select_ip_ic` / `TIEBREAK_IC_CANDLE_TYPE`; `IC_TYPE_PREFERENCE` | ✓ docstrings; `IC_TYPE_PREFERENCE` read by `prefer_candle_type` |
| `pipeline/orchestrator.py` | imports `select_ip_records`, `derive_ip_candidates`; `downstream["ip_candidates"]`, `meta["ip_candidates"]`, `attrs["ip_candidates"]`; the `[ip]` log line | ✓ renamed (`ic_candidates`, `[ic]`) |
| `multitf/types.py` | `LowerTFResult.ip_candidates` | ✓ `ic_candidates` |
| `multitf/entity_df_mutation.py` | `_IP_CANDIDATE_IDX_FIELDS`, the mirror (step 5b), `ip_candidates=down[...]` | ✓ renamed |
| `debug/export_zones.py` | `IP_CANDIDATE_COLUMNS`, `export_ip_candidates` | ✓ `IC_CANDIDATE_COLUMNS` (+ `has_unfilled_imbalance_after`, `is_ic_candidate` after `immediate_imbalance_first_c2_idx`), `export_ic_candidates` |
| `run_replay.py` | `*_ip_candidates.csv` ×2 sites | ✓ `*_ic_candidates.csv` |
| `charting/_zone_render.py` | `ip_rows_by_fib_record(attrs["ip_candidates"])`, the fib hover's `IP: none` | ✓ `ic_rows_by_fib_record`; hover TEXT unchanged in L1 |
| `charting/export_plotly.py`, `export_m15_chart.py` | read `attrs["ip_candidates"]` | ✓ renamed key |
| `zones/fib_tracker.py`, `zones/cross_cycle_fib.py` | docstrings name `select_ip_ic` / `select_ip_records` | ✓ docstrings |
| `structure/structure_engine.py`, MS | `poi_inner_for_fib` (unchanged name) | — (it calls `select_ic` inside) |
| Tests (11 files, ~165 references) | `test_institutional_pattern` 62, `test_plan_h_landing_b` 30, `test_plan_k_landing_a` 19, `test_plan_k_landing_b` 17, `test_fib_chart_render` 15, `test_event_meta_idx_keys` 8, `test_exception_scopes` 4, `test_plan_i_s5_landing_review_pins` 4, `test_e3a_mutation_pins` 2, `test_imbalance_c3_knowability` 2, `test_plan_k_landing_a_review_pins` 2 | ✓ renamed; behaviour pins unchanged (they must pass untouched but for names) |
| Docs | POI_ZONES_SPEC §3 / §3.5, GLOSSARY, ARCHITECTURE, LANDMINES, GOTCHAS, IMBALANCE_FILL_SEMANTICS, MARKET_STRUCTURE_SPEC, CROSS_CYCLE_FIB_SPEC, CHARTING_SPEC, WORKFLOWS ("27 CSVs"), the `/compare` skill (the CSV list + a save-format boundary), the register | ✓ §5 |

**Added by the cold review (§8):**

| Site | L1 |
|---|---|
| `structure/market_structure.py:80`, `multitf/entity_df_mutation.py:1168` (docstrings name `IPSelectionError`) | ✓ |
| `tests/test_exception_scopes.py:16` `BROAD` (the allowlisted re-raise) | ✓ `"ICSelectionError"` — a missed rename weakens the guard silently |
| The 22 `.candidates` reads (`test_institutional_pattern` ×15, `test_plan_k_landing_a:537-539`, `test_plan_k_landing_a_review_pins:151-153`, `test_plan_k_landing_b:56-68`) + `classify_misses` in 4 test files (incl. the string-keyed spy `test_plan_k_landing_a.py:757`) | ✓ CONTENT changes, each rewritten to its purpose: e.g. `test_institutional_pattern.py:528` `[c.ip.ic_idx for c in sel.candidates] == [9]` → `[c.ic_idx for c in sel.ic_candidates] == [9]` (candle 7, missing the touch, is now a check with `pattern` None) |
| `tests/test_plan_k_landing_b.py:41` (pins the 2-argument predicate) | ✓ rewritten deliberately to the `ICCandleCheck.is_ic_candidate` property |
| The `ip_records` kwarg + locals (poi_zones ×18, orchestrator ×8, tests ×15 incl. spies on `k["ip_records"]`), the `ip_row` / `_ip_rows` locals | ✓ `ic_records` / `ic_row` (follow the approved `select_ic_records`) |
| `poi_ip_hover_text` | ✓ `poi_ic_hover_text` (it builds the ranked-IC line too in L2) |
| `_ip_legs`, `_IP_DETAIL_COLUMNS` | kept (the pattern's legs — Step 2's own) |
| Memory `reference_key_files.md` (the code map names `select_ip_records`, `derive_ip_candidates`, `select_ip_ic`, `IPSelectionError`, `is_ic_candidate(...)`) | ✓ same commit (§5) |
| GOTCHAS:2055 (the debug recipe "IC candidates (`touches_zone` True, …)") | ✓ → `is_ic_candidate` True |
| Historical review scripts (`census_h_plugin`, `replay_variant_h`, `mutants_*`, `cmp_tbu.py`, `cmp_k_landing_a/b`) read the old names / `*_ip_candidates.csv` | not edited — they stop working on L1+ outputs; the `/compare` skill states the chain (run them with `CUR` = S_TBU, then `cmp_l1.py`) |

Not renamed (other meanings): MS's `_replay_step_no_patterns`; the wave-candle / reversal `no_pattern` strings in
`test_wave_candles` / `test_ms_reversal_*`.

### 3.3 L1's `/compare` prediction (vs S_TBU)

- 24 of the 27 CSVs byte-equal (structure events, KL, fib, WVMI, triggers, subs, POI zones …).
- The 3 IC-candidate CSVs: renamed `*_ip_candidates.csv` → `*_ic_candidates.csv` (the save's file names → a keyed
  compare script, `plan_h_inputs/cmp_l1.py`): every S_TBU row equal on the shared columns; the new rows = the
  agreed-opposite candles failing condition 3 (all 54 fail condition 3 only — both lenses' review measured) — **54
  over the window's 26 unique POI records** (records, not picks: 8 of them have none); per CSV (subs 3 and 7 sit on
  both lenses — 32 records per-lens): **H1 +5 (236 → 241), confluence +41 (1283 → 1324), counter +31 (332 → 363)**
  — e.g. sub 7 cyc 0 +19 (11 → 30), sub 2 cyc 0 +7, H1 sid 1 cyc 2 +4; 2 new columns after
  `immediate_imbalance_first_c2_idx`; on every old CANDIDATE row `has_unfilled_imbalance_after` True and
  `is_ic_candidate` == (`touches_zone` and `ic_ref_reason` empty); the 8 `IP: none` rows: both new columns empty
  (excluded from those checks, B-m6); on every NEW row `immediate_imbalance_first_c2_idx` empty (no unfilled gap after
  ⇒ none in the immediate window), `is_ic_candidate` False, the pattern columns filled (ICR-11), never picked.
- **Before L1's replay: delete the 3 stale `artifacts/debug/*_ip_candidates.csv`** (the replay does not overwrite
  them — `cmp_save --names-from S_TBU` would compare the stale files as equal and hide the 3 expected MISSING; the
  compare skill's `*_sids.csv` trap). `/commit-save` copies only files newer than its marker (safe).
- Charts: the three figures' FULL figure JSON equal incl. hovers (`cmp_fig_full.py`: exactly 3 MISSING + 3 NEW CSV
  names, every figure equal): H1 81 / 236, counter 135 / 120, confluence 274 / 264.
- The log: `[ic] … rows=` up by the new rows; every other log line equal.

### 3.4 L1 tests (new pins)

`check_ic_conditions`: one candle per failing condition (each outcome on its own, every combination of 2–4 together),
the agreed-direction row set (a split candle not listed, a volume-only doji listed); `find_ic_candidates` = the
all-pass ones; the shared helpers' unit tests (each with two different input lists — ICR-9); `prefer_candle_type`
with and without pinbar; the `immediate_imbalance_candles` rename (a window-2 pin through both readers); the CSV's new
rows + columns (an opposite candle with no unfilled imbalance after → a row, `has_unfilled_imbalance_after` False,
`is_ic_candidate` False, pattern columns set, never picked); `find_ic_variant_candidates` (the 3 V-path sites); **D2 with teeth (A8):** a fixture where a candle failing ONLY
condition 3 (L1) carries a BETTER pattern than the IC candidate's — the drawn pick (`classify_all=True`) == MS's pick
(`classify_all=False`) == the IC candidate; the rewritten `.candidates` / `classify_misses` pins (§3.2).
Mutants `plan_h_inputs/mutants_l1.py`: each condition's outcome, the all-pass predicate, the row set, each shared
helper's input list at its two call sites (Step 2's only, in L1), the CSV columns, the mirror field list.

## 4. Landing 2 — the ranking rules (vs L1's save)

### 4.1 What changes

- **`apply_ranking_rules(df, fib_state, config, ic_candidates) -> (pick | None, ranking)`** — `ic_candidates` = the
  IC candidates' CHECKS (their candle type and immediate gap read from the checks, nothing re-derived — A9) (§1 Step 3), called by
  `select_ic` when Step 2 has no pattern: rule 1 a filter (`candle_type != "pinbar"`), then the shared helpers
  (`filter_immediate_unfilled_imbalance`, `prefer_candle_type(include_pinbar=False)`, `most_recent_before_imbalance`)
  on the ranking rules' own list. `ICSelection.ip_outcome` = `pattern` / `no_pattern` / None (Step 2 never ran);
  `no_ic_reason` = `no_ic_candidate` / `all_ic_candidates_pinbar` / None.
- **`ranking` = `(rule, runner_up_ic_idx, in_play_count, imbalance_first_c2_idx)`** — per deciding rule (ICR-14):
  `exclude_pinbar` → runner-up None (rules 2–4 never run when rule 1 leaves one — no computed runner-up) / the Step 1
  count / None; `immediate_unfilled_imbalance` → the most recent candidate in play failing it / the count
  entering rule 2 / the pick's immediate gap; `ic_candle_type` → the most recent normal / the count entering rule 3 /
  None; `most_recent_before_imbalance` → the most recent other candidate in play / the count entering rule 4 / the
  reference gap; Step 1 found exactly one IC candidate and it is not a pinbar → `ranking_rule` None (ICR-10) and the
  other three None too (the plan's default — A4). Every value an `int()` / `str` (B-m9: `_shift_meta_indices` shifts
  only `isinstance(v, int)` — a numpy int would export slice-local silently).
- **POI meta** (`derive_poi_zones`): `ip_outcome`, `ranking_rule`, `ranking_runner_up_ic_idx`,
  `ranking_in_play_count`, `ranking_imbalance_first_c2_idx` on every POI. **On a ranked IC the POI meta's pattern
  fields are ALL None — `ip_rank`, `ip_leg_candle_counts` and the four `pattern_*_idx` (A1 / B-M3):** `classify_ics`
  leaves a no-pattern candidate its first failing variant's counts / reason / detail (the trial kept the counts on 6
  of the 7 new POIs — `derive_poi_zones` nulls them explicitly). `_ZONE_META_IDX_KEYS` += the two `*_idx` keys
  (`test_event_meta_idx_keys`: the 2 keys into `must_see`, `ranking_in_play_count` into `NON_INDEX_INT_META_KEYS` —
  B-m9). The activation call passes no pattern floor for a ranked IC: `pattern_confirmed_idx=None` → `first_active =
  max(cts_established_idx, ic_idx, lifecycle floor)`; the parameter stays KEYWORD-REQUIRED, typed `Optional[int]`, no
  default (A7 / B-m10 — a future IP caller cannot drop the R1 floor silently; 6 test call sites pass it), and
  `derive_poi_zones` asserts `pick.pattern_confirmed_idx is None` ⇔ `ip_outcome == "no_pattern"`.
- **The debug CSV** (`ICCandidateRecord`, `IC_CANDIDATE_COLUMNS`): `ip_outcome` (record-level, every row of the
  record), `ranking_rule` / `ranking_runner_up_ic_idx` / `ranking_in_play_count` / `ranking_imbalance_first_c2_idx`
  (the picked row); the no-IC row's `reason` `no_ic_candidate` / `all_ic_candidates_pinbar`; `no_pattern` stays as
  `ip_outcome`; `_IC_CANDIDATE_IDX_FIELDS` += the two `*_idx` fields (the mirror test enforces it). A ranked IC's
  picked row keeps its pattern DIAGNOSTICS like every other failing row (`reason` = the first failing pattern rule,
  `legs`, the stacks) — §8.2 (c); the drawn == picked-row check (`test_institutional_pattern.py:850-855`) applies to
  pattern picks only.
- **Hovers (ICR-7, ICR-15)** (`poi_ic_hover_text`, `fib_hover_status_text`): the POI hover of a ranked IC `No
  pattern · ranking rule <rule>` (+ ` over <runner-up>` and the gap when the rule has them) — with `ranking_rule`
  None (one candidate): `No pattern · only IC candidate` (§8.2 (a)); NO `IC <idx>` on the POI hover's pattern line —
  its own `ic_idx=` line names the IC (§8.2, 2026-10-10); the fib hover's outcome line `IC: none (<reason>)` (was
  `IP: none`), `IC: not evaluated (no POI record)` (was `IP: not evaluated …`, §8.2 (b)), a ranked pick `No pattern
  · IC <idx> · <ranking rule <rule> | only IC candidate>` (the IC index KEPT — the fib box's only pointer to its
  IC); an IP pick unchanged (`IP <rank> · IC <idx>`).
- **MS**: `poi_inner_for_fib` returns a ranked pick's inner like a pattern pick's (one selector).

### 4.2 Audit (L2) — `poi_zones` (`select_ic`, `apply_ranking_rules`, `derive_poi_zones` meta + activation call,
`derive_ic_candidates`), `_compute_poi_activation_history` (Optional pattern floor), `entity_df_mutation`
(`_ZONE_META_IDX_KEYS`, `_IC_CANDIDATE_IDX_FIELDS`), `export_zones` (columns), `_zone_render` (both hovers),
CHARTING_SPEC / POI_ZONES_SPEC §3–§5 / GLOSSARY, MARKET_STRUCTURE_SPEC "Snapshot vs per-candle — imbalance
knowability" (the ranking rules' imbalance reads fall under the tiebreakers' argument — A10), the `/compare` skill (a
save-format boundary), tests — incl. `test_event_meta_idx_keys` (B-m9), `test_poi_activation_moment.py:270/300/318`
(the activation signature), `test_pooled_structure_build.py:90` (reads the POI pattern meta),
`test_zone_proximity.py:127` (an `ip_rank` fixture), `test_plan_i_s2_cross_fib_limit.py:676` (the `IP: not
evaluated` relabel).

### 4.3 L2's `/compare` prediction (vs L1's save) — the trial replay, measured 2026-10-09

From PLAN_H §8.9 "the trial replay" (the same rules; only the names / format differ):
- Every structure-events / KL / fib / trigger / sub / WVMI CSV byte-equal — MS's 80 changed reads move nothing.
- **6 CSVs differ** (B-m4): the 3 POI CSVs (all 24 existing rows — H1 3, confluence 14, counter 7 — equal but the 5
  new meta keys, a format change; `export_poi_zones` writes the whole meta) and the 3 IC CSVs (new columns; counter
  only the format) → a keyed `plan_h_inputs/cmp_l2.py` (meta per key, CSV columns by name — a positional diff breaks
  on both).
- POIs: **7 new POIs**: H1 **180** (buy, live 430 →
  652 `next_cycle`) and **694** (never live — collapsed cycle); confluence **670** (never live — collapsed), **929**
  (live 1020 → 1224), **1804** (never live — collapsed), **2575** (live 2608 → 2829 `reversal`), **3809** (live 3819
  → 4200 `reversal`); `ranking_rule` 180 / 929 `exclude_pinbar`, 2575 / 3809 `immediate_unfilled_imbalance`, 694 /
  670 / 1804 `most_recent_before_imbalance`; their pattern meta all None (§4.1 — the trial kept the failing counts on
  6 of them: 180 '1-2-2', 694 '2-1-1', 670 '3-1-1', 1804 '3-3-3', 2575 '2-2-2', 3809 '1-2-3'; 929 None).
- The IC CSVs: those 7 records' no-IC rows → picked rows; sub 1 cyc 2 keeps its no-IC row (`no_ic_candidate`); the
  new columns.
- Charts: H1 traces 81 → 83 / shapes 236 → 239, confluence 274 → 282 / 264 → 276, counter 135 → 137 / 120 → 123
  (the H1 overlay); hovers: the 7 POIs, their fibs' outcome lines, every `IP: none` → `IC: none`.

### 4.4 L2 tests + mutants

**The existing pins L2 flips — MEASURED (B-M2): the full suite on the trial patch (`ic_ranking_trial_patch.py` on a
`git archive` of `92b8b61`), 9 failed / 1528 passed (62.7 s):**

| Test | Why it flips | L2 |
|---|---|---|
| `test_plan_h_landing_b::test_a_record_without_a_pattern_draws_no_poi` | a no-pattern record now draws a ranked POI | rewrite: the ranked POI drawn; + a no-IC record (all pinbars / no candidate) draws none |
| `test_plan_h_landing_b::test_resolver_returns_nothing_when_no_candidate_has_a_pattern` | MS's read returns the ranked IC's inner | rewrite: the ranked inner; + all pinbars → None |
| `test_plan_h_landing_b::test_h1_chart_hovers_show_the_pattern_and_ip_none` | the hover text | the L2 texts |
| `test_fib_chart_render::test_h1_chart_hides_a_collapsed_cycles_fib_and_hovers_the_lifecycle_and_ip` | the fib hover outcome line | the L2 texts |
| `test_institutional_pattern::test_downstream_pipeline_returns_the_selection_of_its_own_fibs` | the none-row reasons; the drawn == picked-row check | the ranked rows; the check on pattern picks only |
| `test_plan_k_landing_a::test_the_reason_is_reported_and_a_cut_candidate_is_never_picked` | `sel.pick is None` — candle 36 is a no-pattern IC candidate → ranked | the ranked pick 36; the cut candidate still never picked |
| `test_plan_k_landing_b::test_ic_candidates_without_a_pattern_stay_no_pattern` | its premise | rewrite: `ip_outcome == "no_pattern"` + the ranked pick |
| `test_ms_reversal_terminal::test_a_reversal_inside_a_post_apply_range_backfill_does_not_finalize_the_range[1]` / `[-1]` | **MS-level** — checked per GOTCHAS ("compare MS's POI reads before touching MS"): exactly ONE MS read differs — at candle 10, cycle 1's fib 7..9: HEAD `None` (no pattern → BOS inner only), the trial `0.6107` (a ranked IC's inner) → the sd-proximity range starts at 11, not 12. The test's subject — ONE terminal at 13 (`range`) — still holds; no MS code involved | make the fixture independent of the IC rules: `_run(..., poi_inner_for_fib=<no POI>)` — its subject is the terminal, its RANGE_STARTED-at-12 premise was built on "no POI" (the alternative: re-pin 11 with the reason) |

Not flipped by the trial but flipping in the real L2 (format the trial did not add): the POI meta key order
(`test_plan_h_landing_b.py:321-327` — 5 new keys), `test_plan_i_s2_cross_fib_limit.py:676` (`IP: not evaluated` →
`IC: not evaluated`), the `IP: none` hover pins (`test_plan_h_landing_b.py:413-447`, `test_fib_chart_render.py:120-126,
:260`).

**New pins:**

Per ranking rule: the decider (one left after it) and the pass-through (none / two or more); ICR-1 (neighbours
sharing one gap → the more recent); ICR-8 (the reference after the oldest IN PLAY — the sub 0 cyc 0 shape: a gap
before every maru in play must not decide); ICR-10 (one non-pinbar candidate → `ranking_rule` None); all pinbars →
no IC + `all_ic_candidates_pinbar`; the H1 sid 0 cyc 0 shape (a pinbar with the only immediate gap is excluded first
→ the maru); Step 3 never runs when a pattern exists (a pattern + a better-ranked non-pattern candidate → the pattern);
the tiebreak and ranking rules given different lists (ICR-9); MS's inner from a ranked pick (a fixture); the
activation floor without the pattern term; the POI meta + mirror shift of the two keys; the CSV columns; both hovers.
Mutants `plan_h_inputs/mutants_l2.py`: each rule's filter / order / reference, the pinbar flag at both call sites,
`ip_outcome`, each `ranking_*` value, the floor, the mirror keys, the hovers.

## 5. Docs (each landing, the same commit — memory `feedback_implement_against_docs.md`)

- **L1**: POI_ZONES_SPEC §3 rewritten in the user's steps (Step 1 = today's §3.1 + §3.2 + the touch, Step 2 = §3.4,
  Step 3 "L2", Step 4 deferred) and §3.5 (the CSV: name, rows, columns); GLOSSARY ("IC candidate" = all four
  conditions; "tiebreak rules"; the CSV); the name mentions in ARCHITECTURE, LANDMINES, GOTCHAS,
  IMBALANCE_FILL_SEMANTICS, MARKET_STRUCTURE_SPEC, CROSS_CYCLE_FIB_SPEC, CHARTING_SPEC, WORKFLOWS; the `/compare`
  skill (the CSV list, a save-format boundary "S_L1"); the register; PLAN_H §8.9 pointer.
- **L2**: POI_ZONES_SPEC §3 Step 3 + §4 (the activation floor without the pattern term) + §5 (hovers); GLOSSARY
  ("ranking rules", `ip_outcome`); CHARTING_SPEC (the hovers); the `/compare` skill (boundary "S_L2"); register §D.1;
  MEMORY.

## 6. The names beyond those agreed in ICR-3 — APPROVED (ICR-13: "Ok")

Agreed: `*_ic_candidates.csv`, `find_ic_candidates`, `select_institutional_pattern`, `apply_tiebreak_rules`,
`apply_ranking_rules`, `select_ic`, `ip_outcome`, the `ranking_*` fields, `all_ic_candidates_pinbar`, "IC: none",
`immediate_imbalance_candles`. Proposed for the rest (internal, follow the agreed ones):

| Today | Proposed | Why |
|---|---|---|
| `IPSelection` | `ICSelection` | the answer of `select_ic` (Steps 1–3), not only the IP |
| `IPCandidate` | `ICCandleCheck` | one checked candle: its condition outcomes + its pattern |
| (new) | `check_ic_conditions` | Step 1's per-candle checks (all four outcomes); `find_ic_candidates` = the all-pass ones |
| `IPRecordSelection` / `select_ip_records` | `ICRecordSelection` / `select_ic_records` | per POI record |
| `IPCandidateRecord` / `derive_ip_candidates` / `export_ip_candidates` | `ICCandidateRecord` / `derive_ic_candidates` / `export_ic_candidates` | the CSV's rows |
| `attrs["ip_candidates"]`, `LowerTFResult.ip_candidates`, `_IP_CANDIDATE_IDX_FIELDS`, `ip_rows_by_fib_record` | `ic_candidates`, `_IC_CANDIDATE_IDX_FIELDS`, `ic_rows_by_fib_record` | follow the CSV |
| `IPSelectionError` | `ICSelectionError` | follows `select_ic` |
| today's `find_ic_candidates` (direction + condition 3; the V path) | `find_ic_variant_candidates` | frees the agreed name; kept with V30 / V60 / V90 (ICR-5) |
| (new) | `filter_immediate_unfilled_imbalance`, `prefer_candle_type`, `most_recent_before_imbalance` | the shared rule helpers (ICR-9) |
| (new CSV columns) | `has_unfilled_imbalance_after`, `is_ic_candidate` | condition 3's outcome; all four pass |

Kept: `ip_rank`, `ip_leg_candle_counts`, `pattern_*`, the `ip_tiebreak` / `ip_tb_*` / `ip_tied_count` fields (Step 2's
own), `TIEBREAK_*`, `IC_TYPE_PREFERENCE`, `poi_inner_for_fib`.

## 7. Size, cost, review

- L1: ~10 engine files (mostly names + the Step 1 / Step 2 split), 11 test files (~165 references) + new pins, ~12
  docs. L2: `poi_zones` + 4 consumers, tests, 4 docs. Each: one replay (~45 s), the mutant runs (full suite each,
  CPU only).
- Reviews (memory `feedback_cold_review_plans.md`, token rule `feedback_token_budget.md`): **this plan** — a cold
  review, 2 read-only lenses (correctness of the rules vs the user's text; the audit's completeness), est. ~0.5M
  tokens; **each landing** — a landing review, 2 lenses, est. ~0.5M each. Total est. ~1.5M (< the ~2M ask line).

## 8. Cold review (2026-10-09) — 2 read-only lenses, 501,249 tokens (est. ~0.5M)

Lens A (rule fidelity + predictions, 223,711) and lens B (audit completeness + landing mechanics, 277,538), code at
`92b8b61`. **0 BLOCKER.** Lens A re-measured both predictions on HEAD and the trial outputs (the 54 rows: all fail
condition 3 only; the 7 picks, their rules and live periods; the MS breakdown 16 / 8 / 1 / 55, 39 all-pinbar);
verified the gating, the separate rule sets (no mixing), rule 4's reference always exists and never straddles, the
trial patch's fidelity, every ICR row covered. Lens B verified byte-identity can hold (`classify_candles` row-wise;
the straddle assert holds for every newly checked candle; the charts read only picked / none rows; the sub mirror is
None-safe and its test enforces the idx-field list; no other consumer reads the IP fields; the fail-loud scope kept).

### 8.1 Findings → folds

| Finding | Sev | Folded into |
|---|---|---|
| A1 / B-M3 — a ranked POI's `ip_leg_candle_counts` (+ the CSV row's diagnostics) ambiguous; the trial kept the failing counts | MAJOR | §4.1 (POI meta all None, explicit), §4.3 (6 rows vs the trial), §8.2 (c) |
| B-M1 — L1 changes what the candidate list CONTAINS (`checks` = every agreed-opposite candle); field / kwarg renames missing; 22 `.candidates` reads + `classify_misses` pins change content | MAJOR | §3.1 (`ICCandleCheck`, `checks` semantics, `ic_candidates` view), §3.2 (the sites) |
| B-M2 — the existing pins L2 flips are not listed | MAJOR | §4.4 — MEASURED: 9 fail on the trial patch (7 expected premise / text pins + the MS fixture, whose one changed POI read was traced) + the format pins listed |
| A2 / B-m5 — per-lens row counts; "26 drawn records" mislabelled | MINOR | §3.3 (exact per CSV) |
| A3 / B-m4 — counter POI CSV changes too (format); 6 CSVs; keyed `cmp_l2.py` | MINOR | §4.3 |
| A4 — ICR-10 wording; the other `ranking_*` None = the plan's default | MINOR | §1, §4.1 |
| A5 / B-m8 — hover text for a ranked pick with no rule; `IP: not evaluated` | MINOR | §4.1, §8.2 (a) (b) |
| A6 — `no_ic_reason` on the early return | MINOR | §3.1 (`no_ic_candidate`) |
| A7 / B-m10 — the activation floor parameter keyword-required, the ⇔ assert | MINOR | §4.1 |
| A8 — D2 pins need teeth (a non-candidate with a better pattern) | MINOR | §3.4, §4.4 |
| A9 — `apply_ranking_rules` must take the checks (no re-derivation) | MINOR | §3.1, §4.1 |
| A10 — MARKET_STRUCTURE_SPEC knowability section | MINOR | §4.2, §5 |
| A11 / B-m8 — missed sites (`market_structure.py:80`, `entity_df_mutation.py:1168`, test files, memory code map, GOTCHAS:2055, `ip_records`, `poi_ip_hover_text`, historical scripts) | MINOR | §3.2, §4.2, §5 |
| B-m6 — `cmp_l1`'s old-row checks fail on the 8 `IP: none` rows | MINOR | §3.3 |
| B-m7 — the 3 stale `*_ip_candidates.csv` in `artifacts/debug`; `cmp_fig_full.py`; the chain to older boundary scripts | MINOR | §3.3, §3.2 |
| B-m9 — guard-test edits; `int()` casts; `test_exception_scopes.BROAD` | MINOR | §3.2, §4.1 |
| B-m10 — invariants (frame read before the guard; the `idx not in df.index` skip; MS's imbalance scans unchanged) | MINOR | §3.1 |

### 8.2 For the user — three labels (new persisted text; proposals)

- **(a)** A ranked pick where nothing needed ranking (one IC candidate, not a pinbar): POI / fib hover `No pattern ·
  IC <idx> · only IC candidate`.
- **(b)** The fib hover's third outcome (no POI record — e.g. sub 3 cyc 3): `IC: not evaluated (no POI record)` (was
  `IP: not evaluated …`), consistent with `IC: none`.
- **(c)** The debug CSV's picked row of a ranked IC keeps its pattern diagnostics (the first failing pattern rule in
  `reason`, `legs`, the stacks) like every other failing row — `ip_outcome = no_pattern` + `ranking_rule` say why it
  was picked; the alternative: blank them on that row.

**Decided 2026-10-10** (the user: "Ok for a) … everything else sounds good"): (b) and (c) as proposed; (a) the
`only IC candidate` wording as proposed. **The IC index on the hovers** (the user: "I'm assuming
the final IC idx is logged elsewhere? In that case I don't think it's useful to log IC <idx> for when the idx is
selected via IP, instead IP <rank> is more information, unless that's now being logged elsewhere?"). Measured on the
L1 code: every POI hover has its own `ic_idx=` line (the 4 POI hover templates, `export_plotly.py:2500,2529`,
`export_m15_chart.py:1670,2627`) and its pattern line (`poi_ic_hover_text`, `IP <rank> · legs …`) carries no
`IC <idx>`; the fib hover's status cell `IP <rank> · IC <idx>` is the only place the fib box names its IC (the fib
box and the POI box are separate hovers); both CSVs carry `ic_idx` (the POI CSV's first column; the IC CSV with
`picked`) and `ip_rank`. **Decided** (the user: "Option 1, keep it on the fib hover."): the POI hover's ranked line
drops `IC <idx>` (`No pattern · <ranking rule … | only IC candidate>` — "No pattern" in the place of `IP <rank>`);
the fib hover keeps it for both pick kinds (`IP <rank> · IC <idx>` unchanged, `No pattern · IC <idx> · …`). §4.1
amended; PLAN_H §6 ICR-15.

## 9. Landing 1 — as built (2026-10-09, uncommitted until the user's go)

- **Code** (`zones/poi_zones.py`, §3.1): `ICCandleCheck` (+ the property `is_ic_candidate`), `check_ic_conditions`,
  `find_ic_candidates`, `_has_unfilled_imbalance_after` (condition 3, shared with the V path), `find_ic_variant_candidates`
  (the V path's finder), the shared helpers `filter_immediate_unfilled_imbalance` / `prefer_candle_type` /
  `most_recent_before_imbalance`, `apply_tiebreak_rules`, `select_institutional_pattern`, `ICSelection` (`checks`,
  `pick`, `tiebreak` + the view `ic_candidates`; L2 adds `ip_outcome` / `ranking` / `no_ic_reason`), `select_ic`;
  the §6 renames across 24 engine + test files (312 occurrences, `plans/` untouched); the CSV's 2 columns
  (`export_zones.IC_CANDIDATE_COLUMNS`).
- **Tests**: 11 files renamed; content rewrites to purpose — the default selections read `sel.ic_candidates` (= the old
  `candidates`), `classify_all` selections read `sel.checks`; `test_plan_k_landing_b`'s predicate test → the property
  (7 cases incl. condition 3); `test_no_candidates_row` → the new listing (7 / 9 / 11 failing condition 3 + the
  none row). New `tests/test_plan_l_landing_1.py` (12 pins: each condition on its own, the row set, the shared helpers
  on two input lists, the window parameter, D2 with teeth, the CSV rows / columns / none row, the CTS-anchor check, the
  type runner-up, the V finder's condition 3). **Suite 1537 → 1551, all pass.**
- **`/compare` vs S_TBU == the prediction (§3.3)**: replay 78.6 s wall (multi_tf_dual 16.0 s, charts 58.7 s — the
  untouched charts slowed alike: machine / OneDrive variance), FETCH GATE PASS (4228 M15 candles); the 3 stale
  `artifacts/debug/*_ip_candidates.csv` deleted first (byte-equal to S_TBU's); `cmp_save`: exactly 3 MISSING, 24 CSVs
  equal, figures equal; `cmp_fig_full.py`: 24 / 24 md5-equal, 3 MISSING + 3 NEW, every figure's FULL JSON equal (H1
  81 traces, confluence 274, counter 135; layouts equal); `plan_h_inputs/cmp_l1.py`: RESULT OK — H1 236 → 241 (+5),
  confluence 1283 → 1324 (+41), counter 332 → 363 (+31), every old row equal, the new columns as predicted.
- **Mutants** (`plan_h_inputs/mutants_l1.py`, full suite each): 25 / 28 KILLED on the first run; the 3 survivors were
  test gaps — L05 (the window ending before the CTS anchor: a CTS-anchor opposite candle is a check that can never be
  a candidate), L20 (the candle-type runner-up), L28 (the V finder's condition 3) — each pinned → **28 / 28 KILLED**.
- **Docs**: POI_ZONES_SPEC §3 in the user's steps (§3.1 with the user's condition numbering), §3.2 / §3.4 / §3.5 / the
  Files table; GLOSSARY (IC candidate, tiebreak rules, IP); the renames in ARCHITECTURE, LANDMINES, GOTCHAS,
  IMBALANCE_FILL_SEMANTICS, MARKET_STRUCTURE_SPEC, CROSS_CYCLE_FIB_SPEC (its retired-read appendix kept), CHARTING_SPEC,
  WORKFLOWS; the `/compare` skill (the CSV list + the boundary "PLAN_L landing 1"); memory `reference_key_files.md`.
- **Landing review (2026-10-09; 2 read-only lenses, 455,967 tokens, est. ~0.5M): 0 BLOCKER, 2 MAJOR (docs), 17 MINOR —
  all folded.** Code + tests (227,427): the selection identical to HEAD by reading AND by a differential fuzz (old
  `select_ip_ic` vs new `select_ic`, both classify modes, 8,000 comparisons, 0 mismatches — tie-rich / gap / reference /
  locked-inactive / CTS-anchor / BOS-anchor motifs); 4 MINOR: the active-or-locked guard now has ONE holder and no pin
  (its ad-hoc mutants R08 / R10 survived — R10 a real MS-path change) → pinned for both consumers; tiebreak rule 3's
  runner-up unpinned (R02 survived; predates L1, `mutants_tbu.py` never mutated it) → pinned; `ICSelection` vs §3.1 →
  §3.1 amended (the L2 fields marked); `find_ic_candidates` off the production path → `select_ic` now uses it (L14
  killed by 17 tests instead of 1); nit `new_ip` → `new_ic`. Docs vs code (228,540): MAJOR — CHARTING_SPEC's
  `"ip_candidates"` key, POI_ZONES_SPEC's "Data Flow" in the old model; MINOR — "condition 4" meaning the reference
  in two places (ARCHITECTURE, a `poi_zones` comment), "before the touch", the IMBALANCE_FILL_SEMANTICS site, the stale
  register pointer ("No-pattern cycles"), Step 3 described as built (register, GLOSSARY), the register status, "§3.1
  candidate" wording, the IC "among the candidates touching", the compare skill's 24 / 27 sentence + an orphaned
  `*_wvmi.csv` row (pre-existing), the code map's "tiebreaks", a stale docstring claim, the GLOSSARY CSV entry,
  PLAN_L §3.1's fields — each fixed. Memory: the Plan E file's boundary list gets S_L1 at `/commit-save`.
- **After the folds**: suite **1553**, all pass; `mutants_l1.py` **31 / 31 KILLED** (+ R08, R10, R02; L14 re-run);
  a fresh replay of the final code (94.0 s wall — run alongside the mutant runner) — FETCH GATE PASS; `cmp_fig_full.py`
  24 / 24 md5-equal + 3 MISSING + 3 NEW, every figure equal; `cmp_l1.py` RESULT OK.
- **COMMITTED `b5c3613` + SAVED S_L1 `20261010_114846_b5c3613` (2026-10-10, the user's go: "everything else sounds
  good")** — save commit `09dffad`, trunk `02a40a7`; a run-mode replay 45.2 s wall (multi_tf_dual 8.8 s, charts
  32.8 s), FETCH GATE PASS; re-checked before the copy: `cmp_fig_full.py` vs S_TBU 24 / 24 md5-equal + 3 MISSING +
  3 NEW, every figure equal; `cmp_l1.py` RESULT OK. = `LATEST_<branch>`, L2's baseline. Local, NOT pushed.


## 10. Landing 2 — as built (2026-10-10, in progress)

- **The flip census re-measured on the L1 code** (step 1 of the session order): `plan_h_inputs/l2_flip_census_patch.py`
  (the ranking rules' behaviour only, no format / names) on a `git archive` of `d6e7db2`, the full suite: **9 failed /
  1544 passed (124 s)** — exactly §4.4's 9 (the trial's, at `92b8b61`); the MS fixture's failure is the same cause
  (`RANGE_STARTED` at 11, not 12). The format pins (§4.4 "Not flipped by the trial"), located on the L1 code:
  `test_plan_h_landing_b.py:321-327` (the POI meta key order), `test_fib_chart_render.py:119-126, :260` and
  `test_plan_i_s2_cross_fib_limit.py:676` (the `IP: none` / `IP: not evaluated` texts); plus fixtures that build a
  no-IC row with `reason="no_pattern"` (`test_fib_chart_render.py:102-111, :214`, `test_plan_k_landing_a.py:549`) —
  they pass, but `no_pattern` is no longer a no-IC reason, so they move to the real reasons.
- **Code** (§4.1): `poi_zones.apply_ranking_rules` (the checks in; rule 1 a filter on `candle_type`, then the shared
  helpers on its own list; `ranking` None when nothing was ranked — ICR-10); `ICSelection` += `ip_outcome` / `ranking`
  / `no_ic_reason` (keyword-required, no defaults) + a `__post_init__` that asserts their consistency (a pick ⇔ no
  reason; a tiebreak only with a pattern; a ranking only on a ranked pick; `no_ic_candidate` ⇔ Step 2 never ran);
  the constants `RANKING_*` (their own, the tiebreak words for rules 2–4 — ICR-S), `IP_OUTCOME_*`, `NO_IC_CANDIDATE`,
  `ALL_IC_CANDIDATES_PINBAR`; `select_ic` runs Step 3 on no pattern; `derive_poi_zones` — the ⇔ assert, the pattern
  meta nulled on a ranked IC, the 5 new keys (`ip_outcome` before `ip_rank`, the `ranking_*` after
  `ip_tb_imbalance_first_c2_idx`), `pattern_confirmed_idx=None` for a ranked IC; `_compute_poi_activation_history`'s
  `pattern_confirmed_idx: Optional[int]` (keyword-required, no default); `ICCandidateRecord` += the 5 fields,
  `derive_ic_candidates` (the no-IC row's `reason` = `no_ic_reason`, `ip_outcome` on every row);
  `entity_df_mutation` (the two `*_idx` in `_ZONE_META_IDX_KEYS` + `_IC_CANDIDATE_IDX_FIELDS`); `export_zones` (the
  columns); `_zone_render` (both hovers; the comments in `export_plotly` / `export_m15_chart`).
- **The ranked POI hover's exact format** (§4.1 named the parts, not the punctuation): `No pattern · ranking rule
  <rule>` + ` over <runner-up>` + ` (gap <c2>)` — `exclude_pinbar` neither, `ic_candle_type` the runner-up only — or
  `No pattern · only IC candidate` (ICR-3's example `… immediate_unfilled_imbalance (gap 2577)` + §4.1's "over
  <runner-up>"); to confirm at the chart review.
- **Tests**: the 9 flips rewritten to their L2 meaning (the M1 frame → the ranked 36 / all pinbars → None; the
  multicycle fixture's cycles 1 / 3 → ranked "only IC candidate"; the hover texts; the drawn == picked-row check on
  pattern picks only); the MS fixture made independent of the IC rules by stubbing `se.poi_inner_for_fib` to "no POI"
  (`monkeypatch` — the suite's precedent, `test_plan_i_s5_landing_review_pins.py:183`; `compute_bounded_structure` has
  no `poi_inner_for_fib` kwarg, so §4.4's `_run(..., poi_inner_for_fib=…)` was not available); the format pins; the
  fixtures that built a no-IC row with `no_pattern` moved to the real reasons, a picked row now needs `ip_outcome`;
  `test_event_meta_idx_keys` (`must_see` + `NON_INDEX_INT_META_KEYS`). New `tests/test_plan_l_landing_2.py` (46 pins on
  real classified frames, + 2 after the mutant run = 48: each ranking rule deciding + passing through, ICR-1 / ICR-8 / ICR-10, all pinbars, the H1
  sid 0 cyc 0 shape vs the tiebreak order, Step 3 not called with a pattern, rule 3's runner-up, rule 4 over rule 3's
  survivors, a non-candidate unseen, the helpers' input lists, MS's ranked inner, the floor, the meta nulling + types,
  the ⇔ assert, the mirror, the CSV, the hovers, `ICSelection`'s asserts; + rule 2's runner-up over three candidates
  after the mutant run). **Suite 1553 → 1605, all pass.**
- **Mutants** (`plan_h_inputs/mutants_l2.py`, full suite each; the equivalent ones listed in its docstring): 51 / 52
  KILLED on the first run; the survivor N09 (rule 2's runner-up `max` → `min`) was a test gap — every rule-2 fixture
  had two candidates in play — pinned with `one_immediate_of_three` → **52 / 52 KILLED**.
- **Replay + `/compare` vs S_L1 == §4.3 exactly** (2026-10-10): replay 74.4 s wall (multi_tf_dual 11.9 s, the charts
  57.3 s), FETCH GATE PASS (4228 M15 candles), the silent-skip grep = the window's 4 known unresolved triggers; `[ic]`
  H1 5 records / 5 picks / 239 rows. `plan_h_inputs/cmp_l2.py` (+ `--trial`) RESULT OK: the 21 other CSVs md5-equal;
  the POI CSVs' old rows equal but the 5 new meta keys; the 7 new POIs — H1 180 (`exclude_pinbar`, 3 in play; live
  430 → 652 `next_cycle`), 694 (`most_recent_before_imbalance` over 691, 2 in play, gap 702; never live); confluence 670
  (`most_recent_before_imbalance` over 678, 4, gap 673; never live), 929 (`exclude_pinbar`, 2; 1020 → 1224), 1804
  (`most_recent_before_imbalance` over 1802, 2, gap 1815; never live), 2575 (`immediate_unfilled_imbalance` over 2574,
  2, gap 2577; 2608 → 2829 `reversal`), 3809 (`immediate_unfilled_imbalance` over 3806, 5, gap 3810; 3819 → 4200
  `reversal`); the IC CSVs H1 241 → 239, confluence 1324 → 1319 (the 7 records' no-IC `no_pattern` rows gone, each record's
  ranked candidate row now its picked row),
  counter 363 = 363 (format only), sub 1 cyc 2 keeps `no_ic_candidate`; the trial's picks / rules identical; figures H1
  81 → 83 / 236 → 239, confluence 274 → 282 / 264 → 276, counter 135 → 137 / 120 → 123, the layouts equal but the
  shapes, the keyed trace diff only POI / fib-hover traces (the 3 never-live POIs are collapsed cycles — hidden, their
  fibs too). The changed hover texts (`plan_h_inputs/l2_hover_diff.py`): POI `No pattern · ranking rule
  exclude_pinbar` (180, 929), `… immediate_unfilled_imbalance over 2574 (gap 2577)` (2575), `… over 3806 (gap 3810)`
  (3809); fib `No pattern · IC 180 · ranking rule exclude_pinbar` (and 929 / 2575 / 3809), `IC: none
  (no_ic_candidate)` (conf sub 1 cyc 2), `IC: not evaluated (no POI record)` (sub 3 cyc 3, both lenses).
- **Landing review (2026-10-10; 2 read-only lenses, 506,850 tokens, est. ~0.5M): 0 BLOCKER, 2 MAJOR (docs), 10 MINOR,
  7 NIT — all folded.** Code + tests (260,433): the rules vs every ICR row verified by reading; a differential fuzz of
  `apply_ranking_rules` vs the trial's `_trial_ranking` (6,300 random classified frames, 3,442 Step-3 comparisons —
  every rule deciding — 0 mismatches on the pick and the rule; the ranking tuple vs an oracle from ICR-14, 0
  mismatches; both classify modes identical on all 6,300); the 7 POIs re-derived from the CSVs; 10 / 12 ad-hoc mutants
  killed — MINOR: the 2 survivors were runner-up gaps (rule 2's runner-up taken from every candidate, a pinbar
  included; rule 3's losers from every non-pinbar, a rule-2 failer included — every fixture had no pinbar candidate and
  no rule-2 filtering before rule 3) → 2 fixtures (`runner_up_not_a_pinbar`, `runner_up_still_in_play`) + mutants N53 /
  N54; MINOR: `ICSelection.__post_init__` accepted three inconsistent states (MS's read never passes `derive_poi_zones`'
  assert) → asserts "a pattern pick has a rank" / "a ranked pick none" + cases (N56 / N57; the third, "all pinbars ⇒
  no_pattern", is implied by those — its assert + mutant N55 dropped as equivalent); MINOR: stale "IP pick" / `IP: none`
  code comments (`poi_zones`, `market_structure`, `fib_tracker`, `orchestrator`, `run_replay`); NIT: the resolver pin
  asserts the `exclude_pinbar` ranking its comment claimed; the collapsed-cycle chart pin's POI shapes pinned exactly
  (5: outline + 2 fills + 2 confirm lines). Docs vs code (246,417): MAJOR — ARCHITECTURE's floor formula and "the IC =
  the institutional-pattern pick"; CROSS_CYCLE_FIB_SPEC §8's MS read (a ranked inner too); MINOR — the `/compare` skill's
  "the figures differ by the 7 POIs" (4 are drawn: 694 / 670 / 1804 sit on collapsed cycles) and "`no_pattern` rows
  become picked rows" (they are removed; a candidate row becomes the pick — also here, above); IMBALANCE_FILL_SEMANTICS'
  read table (+ LANDMINES ×2) without the ranking rules; LANDMINES' "no pattern" (D7) wording (+ a test docstring); the
  GLOSSARY's ambiguous ICR-10 condition; the PLAN_L / register status lines; the memory code map's first clause; NIT —
  `IP: none` → no-IC row (POI_ZONES_SPEC ×2, test comments ×2), CHARTING_SPEC's doubled "IP / IC outcome", POI_ZONES_SPEC
  "set only when a ranking rule decided" → per §3.6, this section's test count.
- **Chart review (2026-10-10): APPROVED** — the user, on the 7 ranked POIs, the hovers and the ranked POI line's
  format (`… ranking rule <rule> over <runner-up> (gap <c2>)`, the parts per rule): "this looks good let's proceed".
- **After the folds**: suite **1609**, all pass; `mutants_l2.py` **56 / 56 KILLED** on the final code (the 52 + N53,
  N54, N56, N57); a fresh replay (201.3 s wall — run alongside the mutant runner), FETCH GATE PASS; `cmp_l2.py`
  (+ `--trial`) RESULT OK.
- **COMMITTED `6d8cc67` + SAVED S_L2 `20261010_155816_6d8cc67` (2026-10-10, the user's go: "this looks good let's proceed")** — save commit
  `8aa4f80`, trunk `48ad744`; a run-mode replay 85.4 s wall (multi_tf_dual 18.4 s, charts 61.8 s), FETCH GATE PASS;
  re-checked before the copy: `cmp_l2.py` vs S_L1 RESULT OK. = `LATEST_<branch>`. Local, NOT pushed.
