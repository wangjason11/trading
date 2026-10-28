# PLAN_U — the rolling pinbar body floor: p90 of the 90 most recent pinbars, the history fetched before the frame (the user's floor calibration, 2026-10-27 / 28)

**Status: U1 LANDED 2026-10-28 (= §4.2; source `b64b10a`, save **S_U1** `20261028_090632_b64b10a`, U2's `/compare`
baseline; §9); the cold review DONE (§8 — no blocker; its U2 amendments carried, §9 "U2"); U2 LANDED 2026-10-28 (= §5.2
exactly; source `84fbbc6`, save **S_U2** `20261028_113834_84fbbc6`, U3's `/compare` baseline; §9 "U2"); U3 the final
window LANDED 2026-10-28 (= §6 exactly — the user's pick after seeing both, natural kept as a switch; source
`32c544d`, save **S_U3** `20261028_123113_32c544d` = the new `LATEST_<branch>`, U3b's `/compare` baseline; §9 "U3"); U3b
NEXT — the floor seed guard steps back until final mode converges (the user's "Guard extends", §9 "U3").** U1's
baseline **S_controls**
`artifacts/commits/week9-entries-decisions/20261027_164527_5dfb5b3` (`LATEST_week9-entries-decisions` until S_U1; source
`5dfb5b3`, HEAD at drafting `4914f31`, code = `5dfb5b3`); suite 2108. Session order item 7 (register
`IDEA_PARKING_LOT.md` §D "SESSION ORDER" = §D.1 item 4). THREE landings, one cause per `/compare`, each with its own
prediction, `/compare` and chart look: §4 U1 the per-candle floor plumbing + its CSV columns + the candle hover, on
TODAY's fixed floor (format only) → §5 U2 the rolling floor, natural window (the behaviour change) → §6 U3 the final
window (a setting flip, a variant replay; the user picks one).

Inputs (`qa_pinbar_floor_inputs/`, run from the repo root with `PYTHONIOENCODING=utf-8`): the measured tables
(canonical in `features/CANDLE_BODY_FLOOR_NOTES.md` "Measured distribution") — `deciles.py`,
`rolling_pinbar_percentiles.py`, `rolling_floor_demotions.py`, `current_floor_demotions.py`,
`floor_history_from_saves.py`, `window_natural_vs_final.py`; THIS plan's prediction `plan_u_prediction.py` (+ `.txt`:
the per-candle list); the seed guard's step counts `seed_guard_steps.py` (H1 2 steps = 559 candles / 232 body-%
pinbars, M15 2 steps = 480 / 189); the cold review's independent re-derivation `review_rederive_predictions.py` (its
own floor loop + the literal guard; = every §5.2 / §6 count — the oracle for U2 / U3: the engine's types must equal
its U2 / U3 columns); U1's checks `u1_landing_check.py`, `m15_hover_floor_count.py`, `u1_hover_example.py` (the
rendered hover example); U2's checks `u2_engine_vs_oracle.py` (the engine's own fetch + classify vs the
prediction's per-candle list), `u2_compare.py` (the downstream keyed per structure vs S_U1), `u2_causes.py` (the
classification deltas in each moved event's window), `gate_floor_seed_sim.py` (the fetch gate's seed branches),
`mutants_u2.py` (+ `u2_fig_named_diff.txt`, `u2_chart_look_times.txt`, `mutants_u2_rerun.txt`); U3's
`make_variant_u3.py` (the variant tree), `u3_engine_vs_oracle.py` (the engine's final-mode types vs the U3 column — run
where the default is "final"), `u3_compare.py` / `u3_causes.py` (the downstream vs S_U2, keyed, and each move's
candles), `u3_changed_candles.py` (every candle U2 ↔ U3, both floors), `u3_fig_detail.py` (+ `u3_fig_named_diff.txt`,
`u3_m15_hover_floor_count.txt`), `u3_window_example.py` (the two windows at H1 446), `u3_convergence.py` (the
convergence check measured on 2 – 5 window seeds); each with its `.txt` where it prints one.

---

## 1. The decisions (verbatim in CANDLE_BODY_FLOOR_NOTES "Direction — the rolling floor")

- **The floor is dynamic** — the user, 2026-10-27: "I want the floor to be dynamic based on 90 most recent pinbar
  candles, and the floor should be the most recent 90 pinbar's p90. We would also need to put in guards to retrieve
  additional historical data if necessary to fulfill this requirement."
- **Both windows are tried** — 2026-10-28: "The one thing I'd like to try though is both using body-% pinbars only and
  also the ones the floor itself demoted. I'd like to see how much they would differ." Every other recommendation of
  the 2026-10-27 reply agreed ("Otherwise, your other recommendations sound good to me"), then the follow-ups (the
  guard of 2 windows, no M5 fetch, the landing order, the review optional): "all these sounds good".
- Agreed rules, specified in §3: the window strictly before the candle; numpy linear interpolation; a body strictly
  below the floor demotes (equal keeps its type); the history fetched BEFORE the frame and dropped (the frame and every
  candle index unchanged); a replay stops with an error when that history can't supply the window, a frame without
  it (tests, short frames) has no floor on the candles without a full window; a stacked leg takes the floor at its LAST
  candle; each candle's floor + a "pinbar by the floor" flag in the final CSV and the candle hover (a rendered example
  first); the fixed per-TF table removed; the window size and percentile are settings.

## 2. Today (code at `5dfb5b3`)

- `features/candle_classifier.py:33` `DEFAULT_PINBAR_BODY_PIP_FLOOR_BY_TF` = H1 2.20 / M15 1.0 / M5 0.5 pips; its one
  reader `candle_params_for_timeframe` (:40) → `CandleParams(pinbar_body_pip_floor=…)` (`features/candle_params.py:32`,
  default 0.0 = off). `apply_candle_classification(df, timeframe="H1")` (:58) classifies with it (:83) and stamps
  `df.attrs["timeframe"]`.
- `features/candles_v2.py::classify_candles` (:91) — step 4 (:151-160): `body_len < floor_pips × pip − EPS` (EPS
  1e-12, :10) → `candle_type = "pinbar"`, `is_special_maru = False`; `pinbar_dir` / `normal_dir` are computed after it.
  The comment cites GOTCHAS "Body-pip floor float precision" — no such entry exists (dangling). `compute_candle_features`
  (:341) = metrics → `classify_candles` → `classify_big_flags` (the prior-5-maru pool, :229) → volume features.
- Classification runs ONCE per frame: H1 `pipeline/orchestrator.py:511` (`apply_candle_classification(df)` inside
  `run_pipeline`, :488); M15 `multitf/data_bridge.py:113` (`prepare_lower_tf_data`, called at `orchestrator.py:1044`
  on `fetch_lower_tf_data(pair, "M15", h1_start, h1_end)` :1043, only when the H1 has triggers). No sub slice is
  reclassified. The replay never builds an M5 frame (`config.lower_timeframes = ("M15",)`).
- The stacked legs (`patterns/institutional_pattern.py`): `_Frame` (:207) holds the frame's arrays + `params`;
  `_stack_types` (:227) classifies every multi-candle stack in ONE `classify_candles(…, fr.params)` call (:235); the
  params come from `poi_zones.py:857` `candle_params_for_timeframe(df.attrs["timeframe"])`.
- The H1 history guard `run_replay.fetch_history_with_auto_extend` (:157; min_history 50, extend_days 4, max 8) is
  about the start extreme and MOVES the frame start (2025-12-01 → 2025-11-15; H1 idx 0 = 2025-11-16 22:00).
- The candle hover: `charting/export_plotly.py:413-446` (H1), `charting/export_m15_chart.py:898-936` (M15) — the
  `TF=` / `idx=` / OHLC / `candle_type=` / `body_len=` … block (not yet on CHARTING_SPEC "Hover label conventions").
- Per-candle CSV: the H1 `*_final.csv` only (`run_replay.export_csv`, every column of `res.df`); there is no
  per-candle M15 CSV.

## 3. The rule

**The floor of candle i** = `numpy.percentile(body_len of the window, pinbar_floor_percentile)` (linear
interpolation, price units), where the window = the `pinbar_floor_window` most recent pinbars STRICTLY before i. No
full window → no floor (NaN). **Step 4 of `classify_candles`:** a candle whose body-% type (after the special-maru
promotion) is maru or normal and whose `body_len < floor − EPS` becomes a pinbar (`is_special_maru = False`, as today);
`pinbar_dir` / `normal_dir` after it, as today.

**Which pinbars fill the window** (`pinbar_floor_window_source`):
- `"natural"` — the body-% pinbars (step 4 not applied): a candle's floor depends only on the candles before it.
- `"final"` — the final pinbars (body-% + the ones step 4 demoted): computed candle by candle in index order, each
  demotion entering the next windows.

Whether a window includes the candle itself would not matter: only a body-% pinbar could be in its own window, and it
is already a pinbar.

**The settings** (`CandleParams`, new fields — origin: the user 2026-10-27 / 28; read sites: the floor function §3.1
and the seed guard §3.2): `pinbar_floor_window: int = 90`, `pinbar_floor_percentile: float = 90.0`,
`pinbar_floor_window_source: str = "natural"` (`"natural"` | `"final"`, anything else raises),
`pinbar_floor_seed_windows: int = 2`. `pinbar_body_pip_floor` and the per-TF table are removed (U2).

### 3.1 The floor function (new `features/pinbar_body_floor.py`, pure)

`rolling_pinbar_body_floor(body_len, body_pct_type, params) -> np.ndarray` — one float per row (price units, NaN =
none), over the rows given (seed + frame, in time order). `pinbar_body_floor_for_frame(frame, seed, params)` = the
function over `seed` + `frame` concatenated, the frame's part returned; `seed` = None → the frame alone. The seed and
the frame are each run through `compute_candle_metrics` + the body-% part of `classify_candles` (no floor); the seed
contributes ONLY to the windows — the frame's prior-maru pool, big flags and volume features stay frame-only (as
today). Asserted: the seed's last candle is strictly before the frame's first.

### 3.2 The history before the frame — the seed guard

`run_replay.fetch_floor_seed(pair, timeframe, first_time, *, min_pinbars, step_candles=400, max_steps=8)`: step k
fetches `[first_time − k × 400 candles of the TF's time, first_time)` (H1 400 h ≈ 16.7 days, M15 100 h ≈ 4.2 days) —
the whole span re-fetched per step, like the auto-extend — until the seed holds ≥ `min_pinbars` body-% pinbars
(`pinbar_floor_seed_windows` × `pinbar_floor_window` = 180). Each request retries like `_get_history_retrying` (same
policy constants) under its OWN prefix `[floor_seed]` — never `[data_bridge]` / `[multi_tf:dual]`, which the fetch
gate's N/A branches key on. One line per TF on success:
`[floor_seed] H1: <n> candles <first> -> <last>, <k> steps, <p> body-% pinbars (need 180)`; after `max_steps` with
fewer → `RuntimeError("[floor_seed] ERROR: …")` (the replay stops before its timing block → the fetch gate FAILs).
Measured on this window (2026-10-28, the 400-candle steps): **H1 2 steps, 559 candles, 232 body-% pinbars; M15 2
steps, 480 candles, 189.**

Why 2 windows, not 1: the natural window needs exactly 90 (a seed with 93 H1 pinbars already gives the 60-day
result); the final window depends on where its computation started until that start is far enough back (H1: 4
candles still differ at 15 days / 93 pinbars, 1 at 20 days / 130, 0 at 25 days / 161; M15 0 from 6 days / 159 —
`window_natural_vs_final.txt` §8). 180 covers both here — an empirical margin for the final window (one pair, one
window), harmless for the natural one (its result can't change).

`main()` fetches the H1 seed right after `fetch_history_with_auto_extend` (before the frame's final idx 0) and, when
`"M15" in CONFIG.lower_timeframes`, the M15 seed before the same first time (the M15 frame starts at the H1 frame's
first time) — counted in the `data_fetch` timing — and passes them as `run_pipeline(df, lower_timeframes=…,
floor_seeds={"H1": …, "M15": …})`. The pipeline does no new network call: `run_pipeline` →
`apply_candle_classification(df, floor_seed=floor_seeds["H1"])`; `_run_multi_tf_dual(…, m15_floor_seed=…)` →
`prepare_lower_tf_data(m15_raw, floor_seed=…)` → `apply_candle_classification(…, timeframe="M15", floor_seed=…)`.
`floor_seeds=None` (every test, `debug/zone_proximity_diag.py`) → frame-only floors: candles before the frame's
`pinbar_floor_window`-th pinbar have none.

### 3.3 The stacked legs

`_Frame` also holds the frame's `pinbar_body_floor` column (a KeyError when absent — fail loudly, like `pair`);
`_stack_types` passes each stack the floor at its LAST candle (`floor[b]` for the stack `(a, b)` — the moment the leg
is complete). A single-candle leg keeps reading the frame's final type (`fr.ctype[first]`), as today.

### 3.4 The columns and the hover

New df columns, written by `classify_candles` on every frame (so in the H1 `*_final.csv`, and carried by the M15 lens
dfs and every sub slice): `pinbar_body_floor` (float, price units, NaN = no floor), `candle_type_before_floor`
(maru / normal / pinbar: the body-% type after the special-maru promotion), `is_floor_pinbar` (bool: a pinbar
because of step 4 only). The candle hover (H1 + M15) gains one line after `candle_type=`, on the hover conventions:
`pinbar_body_floor=0.000283`; on a floor pinbar `pinbar_body_floor=0.000283 | candle_type_before_floor=normal`; no
floor `pinbar_body_floor=NA (fewer than 90 prior pinbars)`. A rendered example before it is locked (U1).

## 4. U1 — the per-candle floor plumbing, the columns and the hover, on TODAY's fixed floor (format only)

### 4.1 The change
- `classify_candles(df, params, body_floor=None)`: step 4 reads a per-row floor array (price units; None or NaN = no
  floor) instead of `params.pinbar_body_pip_floor`; writes the three §3.4 columns. The comment's dangling GOTCHAS
  reference replaced by the reason in place (a nominal N-pip body can float-round to N × pip − ~2e-15).
- `compute_candle_features(df, params, *, anchor_shifts, floor_seed=None)` builds the floor array: in U1 the constant
  `params.pinbar_body_pip_floor × pip` (pip from `attrs["pair"]`, as today; 0.0 → None). `floor_seed` is accepted and
  unused until U2.
- §3.3 the stacked legs read the column (= the constant in U1 → the same types).
- §3.4 the hover line, both charts (`export_plotly.py`, `export_m15_chart.py`).
- Debug tools calling `classify_candles(df, CandleParams())` keep the natural types (no floor array → no floor).

### 4.2 Prediction — behaviour IDENTICAL
29 of the 30 saved CSVs byte-equal; the H1 `*_final.csv` = S_controls' + 3 columns (dropping them → byte-equal);
`pinbar_body_floor` = 0.00022 on all 1058 H1 candles (0.0001 on M15); `is_floor_pinbar` True on today's 40 H1
demotions (`current_floor_demotions.txt`: 7 marus + 33 normals) and 142 M15. The figures differ only in the candle
traces' `customdata` + `hovertemplate` (`fig_named_diff.py`); PNGs equal.

### 4.3 Tests and mutants
Rework the floor tests to the array (`test_candles_v2.py:349-409`: demote, disabled, the EPS boundary, the direction,
special-maru reset, the JPY pip) + the three columns; the leg tests (`test_institutional_pattern.py:440`
`test_stacks_use_the_frame_pip_floor`, :951 the JPY stack, :1100 the frame's timeframe params) read the column; a new
pin: a stack takes the floor of its LAST candle (a frame whose floor differs between the stack's first and last
candle). Mutants (`plans/qa_pinbar_floor_inputs/mutants_u1.py`, full suite), each killer sketched first: the stack
reads `floor[a]`; step 4 `<=`; the EPS sign; `is_floor_pinbar` set on a body-% pinbar; the columns written before the
special-maru promotion; the hover's NA branch.

### 4.4 Docs (same commit)
CANDLE_BODY_FLOOR_NOTES "Current logic" (the floor is a per-candle input + the columns); POI_ZONES_SPEC §3.4 (:552 "the
SAME classifier and per-TF params" → the floor at the leg's last candle); CHARTING_SPEC (the candle hover line);
GLOSSARY (`pinbar_body_floor`, `is_floor_pinbar`); the code comments in `candle_params.py`, `candle_classifier.py`.

### 4.5 Chart look
The rendered example: the hover on a floor pinbar, a non-floor candle, on H1 and M15 (screenshots or the HTML); the
user's look before U1's `/commit-save`.

## 5. U2 — the rolling floor, natural window (the behaviour change)

### 5.1 The change
- New `features/pinbar_body_floor.py` (§3.1) and the four `CandleParams` settings (§3); `compute_candle_features`
  builds the array with it (`floor_seed` used).
- `run_replay.fetch_floor_seed` + `main()` (§3.2); `run_pipeline(…, floor_seeds=None)`, `_run_multi_tf_dual(…,
  m15_floor_seed=None)`, `prepare_lower_tf_data(…, floor_seed=None)`, `apply_candle_classification(…, floor_seed=None)`.
- Removed: `DEFAULT_PINBAR_BODY_PIP_FLOOR_BY_TF`, `candle_params_for_timeframe` (`poi_zones.py:857` and
  `apply_candle_classification` read `CandleParams()`; `attrs["timeframe"]` stays — the leg cap reads it),
  `CandleParams.pinbar_body_pip_floor`.
- `debug/candle_size_distribution.py`: its fixed-floor demotion table is retired (the rule it measured is gone — and
  with it the EPS-less comparison); the distribution stats stay. The rolling measurements live in
  `qa_pinbar_floor_inputs/`.
- The scripts under `plans/*_inputs/` that build `CandleParams(pinbar_body_pip_floor=…)` are records of their landed
  plans and stay as they are (they run against their own commits).

### 5.2 Prediction — the candle types (exact, `plan_u_prediction.txt`)
The engine's own computation (raw `body_len` floats, `< floor − EPS`) and the 0.1-pip shadow give the same sets
(asserted); every frame candle has a full window.

| | today → U2 | of which from maru / normal | with a pin direction | types maru / normal / pinbar |
|---|---|---|---|---|
| H1 (1058) | **74 become pinbars, 0 stop** | 15 / 59 | 6 | today 289 / 233 / 536 → **U2 274 / 174 / 610** |
| M15 (4228) | **294 become pinbars, 0 stop** | 91 / 203 | 31 | today 1358 / 920 / 1950 → **U2 1267 / 717 / 2244** |

No candle stops being a pinbar: the rolling floor is never below today's on this window (H1 min 2.30 > 2.2; M15 min
1.02 > 1.0). Every one of these 74 / 294 is listed (idx, time, body-% type, body, floor) in `plan_u_prediction.txt`.

**Downstream (measured at the build, not predicted in detail):** every reader of a candle type or pinbar — the
breakout patterns (`structure_patterns.py`: continuous, double_maru, one_maru_continuous, one_maru_opposite), MS's
range-candidate qualification (`market_structure.py:1083`), the KL base patterns (`kl_zones_v1.py` 2-candle, 1-candle
big tail, 3-candle star), the IC Step 2 tiebreak / Step 3 ranking rule 1 (`exclude_pinbar` → more
`all_ic_candidates_pinbar` cycles), the institutional pattern's leg ranks, wave candles + WVMI (pinbar +
`pinbar_dir`; most new pinbars are directionless — 68 of 74 H1), the big flags (15 H1 / 91 M15 marus leave the
prior-maru pool). Expect CTS / BOS timings, zones, subs and triggers to move; the `/compare` reports them per
structure. The fetch gate gains the two `[floor_seed]` lines (§5.4).

### 5.3 Tests and mutants
The floor function: the window strictly before; linear interpolation; the natural vs final membership (a demoted
candle enters the final window only); no full window → NaN; `floor_seed` = None vs a seed (the seeded frame = the
frame-only one wherever both have a full window — the measured check); the source setting's raise. The guard
(`fetch_floor_seed` with a stubbed `get_history`): the step spans, the stop at ≥ 180, the `[floor_seed]` line, the
error after `max_steps`, the seed strictly before the frame. The wiring: `run_pipeline(floor_seeds=…)` reaches both
classifications (a spy, as `test_plan_i_s4_landing_review_pins.py:168` does), the M15 seed reaches
`prepare_lower_tf_data`. The tests that pinned the fixed table (`test_institutional_pattern.py:64` `H1 = …`, :214
`test_candle_params_for_timeframe_is_the_one_floor_table`, the fixtures that sit near the 2.2-pip floor — PLAN_H §8.3a:
"a floor recalibration — item 4 — makes them fail loudly") rework to explicit floors; every other test that changes is counted
and explained at the build (short test frames run with frame-only floors). Mutants (`mutants_u2.py`): the window
includes the candle; `"final"` treated as `"natural"`; nearest-rank instead of linear; the seed target 1 window; the
seed after the frame; the guard's `>=` → `>`; the frame's prior-maru pool fed by the seed.

### 5.4 Docs (same commit)
CANDLE_BODY_FLOOR_NOTES (the rolling rule as "Current logic", the fixed table moved to "Previous logic", the
direction section's status); the code comments; ARCHITECTURE (the replay's data fetch: the seed); the `/compare` skill
§2b + WORKFLOWS "Post-replay log grep" (the expected `[floor_seed]` lines join the gate — measured at the build);
GLOSSARY "pinbar body floor"; the register (§D item 7, §D.1 item 4); memory `project_pinbar_body_pip_floor.md`,
`reference_candle_size_distribution_tool.md`; the inbox drained.

### 5.5 Chart look
Full `/compare` vs S_controls → pause for the user's look at the charts (where the new pinbars sit, what moved).

## 6. U3 — the final window (a setting flip)

`pinbar_floor_window_source = "final"`, nothing else. A variant replay in a short `%TEMP%` dir, `/compare` vs the U2
save. **Prediction (exact): H1 15 candles change type vs U2 (1 becomes a pinbar, 14 stop being pinbars; by body-% type
5 marus + 10 normals in all); M15 64 (19 become, 45 stop; 27 marus + 37 normals in all); U3 types H1 279 / 182 / 597, M15 1280 / 730 / 2218**
(listed in `plan_u_prediction.txt`). The user compares U2 and U3 on the charts and picks one; the unused mode is then
removed (the setting kept only if the user wants it) — committed with the pick: no behaviour change when natural is
kept; the flip + its `/compare` when final is picked.

## 7. Notes and risks

- **The lag** (measured, `window_natural_vs_final.txt` §4–6): the window reaches back ~191 h on H1 (natural; 159 h
  final), ~53 h on M15 (43 h). Two H1 weeks with the same median body (2.5 pips) got floors of 2.40 (9 % of marus +
  normals demoted) and 3.00 (32 %). The knob is the window size, not the source.
- **p90's size**: ~2.5–3 × today's demotions (H1 normals 12.4 % → 34.6 %). The percentile is one setting; a later
  flip to p85 / p80 is a one-value change with its own `/compare`.
- **Frame-only floors** outside a replay (tests, `debug/zone_proximity_diag.py`): the first ~190 H1 / ~210 M15
  candles have no floor, so a debug run on such a frame classifies those candles like a no-floor run.
- **Live mode**: causal by construction; the seed rule is the same at a live start.
- M5: no frame, no seed; the rolling rule is TF-agnostic, so an M5 frame added later needs no table.

## 8. Reviews and cost

Solo by default. A cold review of this plan is optional (2 read-only agents, about 0.5M tokens) — the user's call; a
landing review of U2 likewise.

**Cold review DONE 2026-10-28** (the user: "sure go ahead" to "the 2-agent review of U2 and U3 in the background
and build U1 now"): A = the rule / floor function / final mode / predictions / tests (~202k tokens), B = the wiring /
call sites / guard / docs / outputs (~282k). **No blocker.** A re-derived every prediction from this plan's literal
text with its own script — all MATCH (§3.2 the seed steps, §4.2, §5.2, §6), and the guard's 2-step seed gives the
60-day seed's floors and types on every frame candle, both modes, both TFs. B verified §2 (classification runs in
exactly 3 engine places — `orchestrator.py:511`, `data_bridge.py:113`, the stacks; no slice / probe / resampled frame
is reclassified), the OANDA semantics (`from` inclusive, `to` exclusive → the seed cannot overlap the frame; the M15
frame starts at the H1 frame's first time), every reader of the H1 `*_final.csv` reads by name. Findings and where
each lands:
- **Taken in U1:** a params floor without `body_floor` raises (A3 / B5 — `classify_candles` no longer silently
  ignores it); the mutant "a disabled floor as 0.0, not None" (A2); the M15 count is checked on the M15 chart hovers
  (A4 — `m15_floor_count`, §9); the `*_final.csv` column insertion recorded as a save-format boundary in the
  `/compare` skill + its "Candle Classification" row (B8); LANDMINES "Why it matters" gains `pinbar_body_floor` (B9);
  the stale `prepare_lower_tf_data` docstring and the big-flag comment (B9). The hover reads the new columns with a
  default (B4 — already so).
- **For U2 (amend §3 / §5 when U2 starts):**
  - ONE shared demotion predicate (e.g. `floor_demotes(body, type_before_floor, floor)`) used by the floor function's
    final mode AND step 4; in final mode `compute_candle_features` asserts `is_floor_pinbar` == the function's own
    mask (A1 / B10). Mutant: the function's predicate without EPS (A measured H1 1 / M15 12 bodies under the floor by
    float noise alone at U2, 2 / 17 at U3).
  - §3's "would not matter" is about TYPES only: the window including the candle changes the `pinbar_body_floor`
    value on a body-% pinbar and a stack ending on one (A2) — that is the mutant's killer. The "seeded = frame-only"
    test holds for `natural` only (A2).
  - More mutants (A2): the seed's demotions not counted in the final window; `pinbar_floor_window` /
    `pinbar_floor_percentile` hard-coded (tests at window 5 — small fixtures); the window missing candle i − 1; no
    `time < first_time` filter (a stub returning a candle AT `first_time`).
  - A test seam for explicit floors (A5 / B3): `compute_candle_features(…, body_floor=…)` / `apply_candle_classification
    (…, body_floor=…)` overrides (the IP fixtures are 6-candle frames — no 90-pinbar window). The 8
    `prepare_lower_tf_data` stubs (`lambda df: df`: `test_lifecycle_sweep_unit.py:1936`, `:1994`,
    `test_plan_i_s2_cross_fib_limit.py:973`, `test_plan_t_t1_probe_parity.py:258`, `test_proximity_overlay.py:617`,
    `test_sub_wvmi_per_sub.py:379`, `:455`, `:523`) → `lambda df, **_k: df` (B1). B's floor-0 stand-in run failed 5
    tests: `test_institutional_pattern.py` :214, :440, the JPY stack, `test_first_failing_variant_is_reported`, and
    `test_plan_i_s5_ms_reads_the_fib.py:157` (real M15 rows); :1100 passes but becomes vacuous → rework (B3).
  - The seed: filter `time < first_time` explicitly, then assert (A7); `test_exception_scopes.py`'s allowlist gains
    the seed retry + LANDMINES "the two OANDA `get_history` retries" → three (B2); assert `step_candles × max_steps ≤
    5000` (B12; 3200); the M15 seed is fetched even in a no-trigger run — accepted, said in §3.2 (B12).
  - The fetch gate (B7): the `[floor_seed]` lines are checked in every branch where the run completed (both N/A
    branches too); the THIRD copy of the snippet — `/commit-save` Step 4b — changes with §2b and WORKFLOWS; §2b's "On
    FAIL" names `RuntimeError: [floor_seed] ERROR …`.
  - Debug tools (B6 / A10): `debug/zone_proximity_diag.py` and `debug/probe_fc_finalize.py` ("the EXACT production
    resolution path") call `fetch_floor_seed`; a GOTCHAS entry next to "Auto-Extend Index Shift": a frame classified
    without the seed gets other candle types.
  - Docs (B9): POI_ZONES_SPEC :592-593 ("the candle params … come from the FRAME"), LANDMINES "Why it matters" (the
    stamp = the leg cap only) + :655-663 (the fetch fail-loud paragraph), IDEA_PARKING_LOT §A :41-45, `poi_zones.py`
    :848 docstring, ARCHITECTURE "Execution model §1 Batch / Replay" (the seed), the `/prepare` skill tree + memory
    `reference_key_files.md` (`features/pinbar_body_floor.py`).
  - §5.2 readers to name at the `/compare` (A10): WVMI `_compute_last_wave_weight` (`wvmi.py:43-46` — a new pdir-0
    pinbar with a big-maru ratio keeps weight 1.0, one with only a big-normal ratio drops 1.0 → 0.7);
    `wave_candles._is_compound_first_wave` (:103-105) never takes a pdir-0 pinbar.
  - The H1 CSV's direct checks: `is_floor_pinbar` U1 40 (7 / 33), **U2 114 (22 / 92), U3 101 (17 / 84)**; M15 on the
    chart hovers: U1 142, U2 436, U3 410 (A4; count them with `qa_pinbar_floor_inputs/m15_hover_floor_count.py`).
  - §7: the frame-only floors leave H1 idx 0–211 / M15 0–220 without a floor (measured; A8).
  - The hover's 6 decimals are exact at (90, p90): the virtual index 89 × 0.9 = 80.1 puts every floor on a 0.01-pip
    grid (1e-6); another window / percentile can need more (B13).
- **For U3:** in final mode a candle's type depends on where the history starts (a different frame start, a live
  start) until the runs converge — natural mode has no such dependence (A6); a per-run convergence check (recompute
  with the seed's first window of pinbars dropped, raise if a frame floor differs) is the user's call at U3.

## 9. Build log

### U1 — LANDED 2026-10-28, = §4.2 (the user's look at the rendered hover example: "looks good go ahead and commit-save")
- **Code:** `candles_v2.py` — `classify_candles(df, params, body_floor=None)` (step 4 on a per-row array; the three
  columns; raises when `params.pinbar_body_pip_floor > 0` and no `body_floor` — review A3), `fixed_body_floor`,
  `_pip_size`; `compute_candle_features` passes `fixed_body_floor`; `institutional_pattern._Frame.floor` +
  `_stack_types` (the floor at the stack's last candle); `export_plotly._candle_floor_hover` + the H1 template slot
  12, M15 slot 14. Deviation from §4.1: `floor_seed` is NOT added to `compute_candle_features` in U1 (it would be an
  unused parameter) — U2 adds it.
- **Tests:** suite 2108 → **2121** (`test_candles_v2.py` +7: the columns, the body-% pinbar flag, no floor, per row,
  the length check, `compute_candle_features`, the params-floor raise; `test_institutional_pattern.py` +1: the stack
  takes its LAST candle's floor; new `test_plan_u_body_floor.py` 5: the frame wiring H1 / M15, the hover lines, the
  hover on both charts). **Mutants `qa_pinbar_floor_inputs/mutants_u1.py` 17 / 17 KILLED**, each by its sketched pin.
- **Replay** (run mode, `run.log`): FETCH GATE PASS; skip grep 33 (= S_controls); timing total 66.76 s.
  `u1_landing_check.py S_controls`: 29 CSVs byte-equal; the H1 `*_final.csv` field-equal with the 3 columns' fields
  removed (1058 rows); `pinbar_body_floor` = 0.00022 on all 1058 (written `0.00022000000000000003`, = `2.2 × 0.0001`),
  `is_floor_pinbar` 40 (7 marus + 33 normals). M15 (both lens charts' hovers): 4228 candles at 0.000100, 142 floor
  pinbars (34 / 108). `fig_named_diff.py`: on all three charts only the `OHLC` + `imbalance:bearish` / `:bullish`
  candle traces differ, in `customdata` + `hovertemplate`; traces / shapes / layout equal; the 3 PNGs byte-equal.
- **Landed:** source `b64b10a`, save `60b2d7e`, trunk `53090ea` = **S_U1** `20261028_090632_b64b10a` (run mode on
  the final code: FETCH GATE PASS, skip grep 33, timing total 120.17 s; `u1_landing_check.py` ALL AS PREDICTED vs
  S_controls; the PNGs byte-equal). Suite 2121. S_U1 is U2's `/compare` baseline; its `*_final.csv` has the 3 columns,
  so U2's diff of that file is positional-safe (same columns).

### U2 — LANDED 2026-10-28, = §5.2 exactly (the user's look: "charts look good, go ahead and commit-save")
- **Code:** new `features/pinbar_body_floor.py` (`rolling_pinbar_body_floor` → `(floor, demoted)`,
  `pinbar_body_floor_for_frame`; the seed asserted strictly before the frame); `candles_v2.floor_demotes` = the ONE
  predicate (step 4 + the final window; §8 A1) — `fixed_body_floor`, `_pip_size` and the params-floor raise removed;
  `compute_candle_features(…, floor_seed=None, body_floor=None)` (the rolling floor; `body_floor` the test seam, §8
  A5; asserts the function's demotions == `is_floor_pinbar` in BOTH modes — stronger than §8's "final mode");
  `CandleParams` the four settings, `pinbar_body_pip_floor` removed; `candle_classifier`: the table and
  `candle_params_for_timeframe` removed, `apply_candle_classification(…, floor_seed=None, body_floor=None)` with
  `CandleParams()`; `poi_zones.select_ic` `CandleParams()` (+ the :848 docstring); `data_bridge.prepare_lower_tf_data(…,
  floor_seed=None)`; `orchestrator.run_pipeline(…, floor_seeds=None)` (asserts the keys ⊆ {H1, M15}) →
  `_run_multi_tf_dual(…, m15_floor_seed=None)`; `run_replay`: `fetch_floor_seed`, `fetch_floor_seeds` (what `main()`
  calls — so the 180 and the M15 condition are unit-testable), `_get_history_retrying(…, prefix=, allow_empty=)`.
  Debug: `zone_proximity_diag.py`, `probe_fc_finalize.py` fetch the seeds; `candle_size_distribution.py`'s demotion
  table retired (its natural classes via an all-NaN `body_floor`).
- **Deviations from §3 / §8:** (1) the seed's requests REUSE `_get_history_retrying` (a `prefix` + `allow_empty`
  parameter) instead of a new retry function — so `test_exception_scopes.py`'s allowlist needs no new entry (B2's
  intent, every broad handler listed, holds); LANDMINES now says "two retry SITES, three uses". (2) An empty seed step
  is not an error (`allow_empty`): it holds no pinbar, the guard steps on.
- **Prediction met** (`qa_pinbar_floor_inputs/u2_engine_vs_oracle.py` — the engine's own fetch + classify vs
  `plan_u_prediction.txt`'s per-candle list, 0 mismatches on both TFs; the oracle `review_rederive_predictions.txt`'s
  U2 column): H1 74 become pinbars, 0 stop (15 / 59, 6 with a direction), 289 / 233 / 536 → 274 / 174 / 610,
  `is_floor_pinbar` 114 (22 / 92); M15 294 (91 / 203, 31), 1358 / 920 / 1950 → 1267 / 717 / 2244, `is_floor_pinbar`
  436 (125 / 311) on both lens charts (`m15_hover_floor_count.py`); every frame candle has a floor (H1 2.30 – 3.50
  pips, M15 from 1.02). The seed lines = `seed_guard_steps.txt` (H1 559 / 232, M15 480 / 189, 2 steps).
- **Replay** (run mode): FETCH GATE PASS + both `[floor_seed]` lines; skip grep 33 (the same lines as S_U1); timing
  total 62.04 s (`data_fetch` 2.60 s with both seeds). The gate snippet (all three copies) checks the seed lines in
  every completed branch — `gate_floor_seed_sim.py`: 10 / 10 cases as expected.
- **Downstream vs S_U1** (`u2_compare.py` + `.txt` keyed per structure; `u2_causes.py` + `.txt` the classification
  deltas in each window; `u2_fig_named_diff.txt`): 21 / 30 CSVs differ; byte-equal: both lens `subs` + `triggers`,
  the unresolved table, the H1 + confluence POI zones, imbalance instances, raw — **no sub, trigger, POI pick or IC
  pick moves** (§5.2 expected subs / triggers might). What moves, each with its cause:
  - H1 sid 0 cycle 0: CTS_CONFIRMED 439 (anchor 430, `one_maru_continuous` from 436) → 485 (anchor 448, from 484):
    candle 439 (normal 2.4 pips, floor 3.03) is a pinbar, the pattern fails; price makes raw CTS highs 447 / 448
    first. With it: the CTS KL zone (confirmed 439 → 485, base 430 'base inside bar' → 448 'base', bottom 0.58132 →
    0.58178; 2 bounds steps fewer → −4 traces / −2 shapes on H1 and on both M15 overlays), the fib's `cts_idx` 430 →
    448 (+ `locked_at`, `ic_ref_price`; its IC record 151 → 157 check rows, the same pick), the WVMI (0,0) LB 430 →
    447, FP 436 → 473 (pullback 0.521 → 0.237), H1 WVMI LP rows 57 → 54, the range / state rows 436 – 485. Cycle 1
    and the reversal unchanged.
  - counter sub 6 cycle 0: CTS_ESTABLISHED / BOS_CONFIRMED 3656 (`double_maru` 3655 – 3656) → 3657
    (`one_maru_continuous` 3656 – 3657): 3655 (maru 1.1, floor 1.30) is a pinbar (its POI meta
    `cts_established_idx`, the fib `activated_at` follow).
  - confluence sub 0 cycle 0: the pattern-path CTS_UPDATEDs 780 / 784 → 783 — no type change there: marus demoted
    earlier left the prior-5-maru pool, the pool's max range rose (0.44 → 1.26 pips at 775), so 774 – 782 lost their
    big flags.
  - confluence sub 0 cycle 3: CTS_CONFIRMED 1748 → 1751 (the same pattern from 1747): the big flags again — 1747
    (maru 6.2) ratio 0.95 → 0.34; with it WVMI (0,3) FP 1747 → 1748 (pullback 1.239 → 0.721).
  - confluence sub 1: cycle 0 CTS_CONFIRMED 1818 → 1833 (1818, maru 1.2 / floor 1.61, a pinbar; 1817 lost its big
    flags) — the range from 1821, WVMI (1,0) FP 1817 → 1828 (3.624 → 2.096); cycle 1 CTS_RECONFIRMED 1904 → 1903
    (1903 a pinbar); cycle 4 the CTS_UPDATED at 2364 `one_maru_continuous` → `double_maru` (2363 lost big_maru).
  - sub 3 cycle 2 (both lenses): the BOS KL zone's base 2914 'no base star' → 2915 'base' (2914 maru 1.3 → pinbar);
    WVMI (3,1) LP 2914's weight 1.0 → 0.7 (pullback 0.912 → 0.638) — `wvmi._compute_last_wave_weight` (§8 A10): a
    new pdir-0 pinbar with only a big-normal ratio; WVMI (3,2) FB 2916 → 2917 (2916, a big-normal normal before,
    is a big-maru pinbar with `pinbar_dir` +1 — against sub 3's sell wave, so
    `wave_candles._is_compound_first_wave` (:103-105) rejects it; measured, `u2_causes` + the frame check).
  - Charts (`fig_named_diff.py`): H1 83 / 234 → 79 / 232, counter 158 / 136 → 154 / 134, confluence 306 / 307 →
    302 / 305 — the H1 CTS zone above; the moved dots, swings, state labels, wave-candle traces, sub 3 c2's BOS zone,
    the WVMI LP marks; the candle hovers (every floor).
- **Tests:** suite 2121 → **2146** (`test_plan_u_rolling_floor.py` new, 26: the floor function, the seed, the guard with a
  stubbed OANDA, the pipeline wiring; reworked: `test_candles_v2.py` the floor tests on explicit floors (−3: the
  params raise, the JPY pip, `fixed_body_floor`; +1 `floor_demotes`), `test_institutional_pattern.py` (the fixture
  frames keep U1's 2.2 / 1.0 pips through the seam — `_FIXTURE_FLOOR_PIPS`; the floor-table test removed; N32
  re-pinned on the frame's column, no longer vacuous — §8 B3), `test_plan_u_body_floor.py` (frame-only + the seam),
  `test_unified_probe._prepare_df(body_floor_pips=)` for the F16 frame (`test_plan_i_s5_ms_reads_the_fib.py`: real M15
  rows pinned under the 2.2-pip floor, 30 rows hold no window — §8 B3's predicted failure), the 8
  `prepare_lower_tf_data` stubs `lambda df, **_k: df`. **Mutants** `mutants_u2.py` (+ `.txt`): first run 28 / 29 —
  `u2_function_no_eps` SURVIVED: its pin used a frame whose float bodies (c − o) put the maru exactly AT its floor, so
  the EPS never mattered; re-pinned on exact arrays + a new pin for the agreement assert (`u2_assert_dropped`, now a
  killable mutant) — the re-run 2 / 2 KILLED (`mutants_u2_rerun.txt`): **30 / 30 in all**, each by its sketched pin.
- **Landed:** source `84fbbc6`, save `ca25478`, trunk `562d237` = **S_U2** `20261028_113834_84fbbc6` (run mode on
  the final code: FETCH GATE PASS + both seed lines, skip grep 33, timing total 110.30 s; `u2_compare.py` on the save
  = the compared replay's output, identical). Suite 2146. At the look the user asked which percentile and whether
  the demoted pinbars fill the window (p90 of 90; natural — they do not, that is U3) and whether any default pip floor
  is left (none in the engine; only the test fixtures pass 2.2 / 1.0 pips through the `body_floor=` seam). One
  procedural slip, undone before the save: the commit-save gate was extracted with a `sed` range that also ran the rest
  of the skill (a bogus save commit + its cherry-pick; both refs moved back) — the skill now says so.

### U3 — the variant MEASURED 2026-10-28, = §6 exactly; the user picked FINAL + the convergence check
- **The pick (2026-10-28):** shown this comparison, the user asked "sorry to clarify, final is with demoted included?"
  (yes — the window = the 90 most recent pinbars of either kind; natural = body-% pinbars only; the example at H1 446,
  `u3_window_example.py` + `.txt`: natural reaches back 239 h, p90 3.30 → pinbar; final holds 17 demoted bodies of
  1.1 – 2.6 pips, reaches back 201 h, p90 2.60 → stays normal), then chose **"Final + convergence check"** (over the
  recommended "natural, remove final": the modes differ by four records here, and natural has no start-dependence).
  The optional U2 landing review: "Skip it". So: the flip (the landing = this variant), then the per-run convergence
  check (§8 "For U3") as its own landing.
- **The convergence check, measured before building** (`u3_convergence.py` + `.txt`): as worded — the seed's first
  window dropped, raise if a frame floor differs — it FAILS on this replay with the 2-window seed (H1 57 frame floors
  differ, idx 0 – 182, 1 type, idx 8; the two runs' windows agree only from frame idx 200; M15 0 differ, agree from
  frame idx 15). With 3 windows (the guard's 3 steps: H1 840 candles / 355 body-% pinbars, M15 816 / 310) it passes on
  both TFs, the windows agree INSIDE the seed (H1 row 582 of 840, M15 721 of 816), and the frame floors equal the
  2-window seed's on every candle (also at 4 / 5 windows) — the 2-window result was already converged, the check just
  could not show it. Offered: (1) the guard extends the seed until the full and the drop-one-window runs agree by the
  seed's last candle (seed-only, so the M15 seed is checked before the M15 frame is fetched; raise after 8 steps);
  (2) the check as worded on a fixed 3-window seed; (3) as worded on 2 windows — raises here. **The user: "Guard
  extends"** (the recommendation). **The natural mode: "Keep as a switch"** — `pinbar_floor_window_source` stays,
  default "final", "natural" selectable with its tests.
- **Landing order** (one cause per `/compare`): U3 the flip (= the variant above); then U3b the guard's convergence
  rule (predicted: no candle changes on this window; only the two `[floor_seed]` lines — the gate's expected lines).
- **The flip:** `candle_params.py` the default "final" (+ its comment), `pinbar_body_floor.py` the docstring (final
  the default, its start-dependence), `test_plan_u_rolling_floor.py::test_the_settings_are_the_users` pins "final"
  (the unit tests keep passing each mode explicitly). The landing replay (run mode): FETCH GATE PASS, the seed lines
  unchanged, skip grep 33 (= S_U2's lines), timing total 65.14 s; **= the variant: 30 / 30 CSVs byte-equal, the
  figures equal (`fig_named_diff.py`: no trace changed, layouts equal), the 3 PNGs byte-equal.** Suite 2146.
- **Landed** (the user's look: "charts look good, go ahead and commit-save"): source `32c544d`, save `704785e`, trunk
  `839c9f2` = **S_U3** `20261028_123113_32c544d` (run mode on the committed code: FETCH GATE PASS + both seed lines,
  skip grep 33, timing total 65.17 s; its 30 CSVs = the variant's; `u3_compare.py` on the save = `u3_compare.txt`).
  S_U3 is U3b's `/compare` baseline.
- **Variant** (`qa_pinbar_floor_inputs/make_variant_u3.py`: a `git archive` of `e5c3b07` in `%TEMP%\u3v`, the
  `CandleParams.pinbar_floor_window_source` default "natural" -> "final", nothing else — the setting's one read site is
  `rolling_pinbar_body_floor`): FETCH GATE PASS, both `[floor_seed]` lines = S_U2's (the guard counts body-% pinbars);
  skip grep 33, the same lines as S_U2 (only the tree path in the FutureWarning lines differs); timing total 85.39 s.
- **Prediction met** (`u3_engine_vs_oracle.py` + `.txt`, run in the variant tree — the engine's own fetch + classify vs
  `plan_u_prediction.txt`'s U3 column, 0 mismatches on both TFs): H1 15 change vs U2 (1 becomes a pinbar, 14 stop; 5
  marus + 10 normals; 1 with a pin direction), types 279 / 182 / 597, `is_floor_pinbar` 101 (17 / 84); M15 64 (19 / 45;
  27 / 37; 5), 1280 / 730 / 2218, `is_floor_pinbar` 410 (112 / 298) — also on both lens charts' hovers
  (`u3_m15_hover_floor_count.txt`). A slip in the first draft of the check, fixed before use: "with a pin direction"
  read the U3 frame's `pinbar_dir`, which is 0 on a candle that STOPS being a pinbar — it now takes the direction the
  candle has as a pinbar (the engine's wick rule, every maru / normal made a floor pinbar), as the prediction does.
- **The floors** (`u3_changed_candles.py` + `.txt`, every changed candle with idx, time, body, both floors): final mode
  LOWERS the floor on most candles — a floor-demoted body is below the floor by definition, so each demotion entering
  the window pulls the next p90 down (a negative feedback). H1 the floor differs on 794 of 1058 candles (U3 − U2 mean
  −0.08 pips, −0.70 … +0.40; lower on 536, higher on 258; median 3.03 → 3.00); M15 on 2777 of 4228 (mean −0.03, −0.51 …
  +0.49; median 1.40 both). 8 of the 64 M15 type changes come from a 0.01-pip floor move (the body on the grid, at the
  floor on one side), none of H1's.
- **Downstream vs S_U2** (`u3_compare.py` + `.txt`, `u3_causes.py` + `.txt`, `u3_fig_named_diff.txt`,
  `u3_fig_detail.py` + `.txt`): 20 / 30 CSVs byte-equal — **no sub, trigger, POI, fib or IC pick moves; no H1 or counter
  event moves**. What moves, each with its cause:
  - H1 WVMI (sid 0, cycle 0): FP 473 → 468 (2025-12-12 15:00 → 10:00), pullback momentum 0.237 → 0.645 (+ the 7 LP rows'
    hovers): 457 and 462 (marus, bodies 3.2 / 3.1 pips; floor U2 3.30, U3 3.03) are marus again, so they re-enter
    the prior-5-maru pool and push out an older maru — the pool's max candle length (high − low) at 458 30 → 17 pips;
    458 – 474 get their big flags back (U2 had removed them —
    §9 "U2", the reverse) and 468 (normal 3.4, big) is the first pullback wave candle.
  - confluence sub 0 cycle 1: 916 (normal 1.7, floor 1.60 → 1.71) and 918 (normal 1.6, 1.51 → 1.71) become pinbars —
    the BOS KL zone's base 916 'no base star' → 917 'base' (top 0.57066 → 0.57065, start 11:00 → 11:15), WVMI (0, 0, 1)
    FB 918 → 920 (breakout momentum 0.113 → 0.221; its 20 LP rows' hovers).
  - confluence sub 1 cycle 4: the CTS_UPDATED at 2364 `double_maru` → `one_maru_continuous` (+ its RANGE_RESET /
    STATE_CHANGED `pat`): 2363 (maru, body 3.4) regains `is_big_maru` (the pool's max candle length 8.6 → 6.4 pips) — S_U1's pattern
    back.
  - The IC check rows (not the picks): `candle_type` / leg types on H1 2, counter 10, confluence 23 rows; `ip_rank` on
    1 / 5 unpicked rows.
  - Charts (`fig_named_diff.py`): counts unchanged — H1 79 / 232, counter 154 / 134, confluence 302 / 305. Changed: the
    candle hovers (the floors), the breakout-pattern markers (`struct:*`: H1 +6 markers, 1 renamed; M15 a few added /
    removed / renamed on both lens charts — the same M15 frame), the H1 FP wave candle (+ its full-height vertical line,
    a layout shape) on all three charts, the confluence KL zone + outline of sub 0 c1, its FB wave candle, the WVMI LP hovers.
