"""PLAN_U U2 mutants — run from the repo root:
    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \
        engine_v2/plans/qa_pinbar_floor_inputs/mutants_u2.py --out %TEMP%/mtU2

Each row: (id, relpath, old, new) — `old` occurs exactly once (text mode, LF anchors). The sketch after each = the
pin expected to kill it (written BEFORE the run; the tests: `test_plan_u_rolling_floor.py` unless named). PLAN_U §5.3 +
the cold review's §8 additions. Not a mutant (equivalent): `<` → `<=` in `floor_demotes` (differs only
on a body EXACTLY at floor - EPS).

A RECORD of the run at U2 (`84fbbc6`). Since U3b (`9c3949d`, the loop's window holds rows) four anchors no longer
match — a re-run reports `u2_window_includes_candle`, `u2_nearest_rank`, `u2_window_hardcoded_90` (now twice) and
`u2_guard_strict_gt` BAD; they are re-anchored and re-run in `mutants_u3b.py` (`u2r_*`, `u3b_guard_count_strict_gt`;
all KILLED — `mutants_u3b.txt`).
"""
PF = "engine_v2/features/pinbar_body_floor.py"
CV = "engine_v2/features/candles_v2.py"
CC = "engine_v2/features/candle_classifier.py"
RR = "engine_v2/run_replay.py"
OR = "engine_v2/pipeline/orchestrator.py"
DB = "engine_v2/multitf/data_bridge.py"

LOOP = (
    "        floor[i] = current\n"
    "        demoted[i] = bool(floor_demotes(body[i], btype[i], floor[i]))\n"
    "        if btype[i] == \"pinbar\" or (source == \"final\" and demoted[i]):\n"
    "            window.append(body[i])\n"
    "            if len(window) == size:\n"
    "                current = float(np.percentile(np.asarray(window), pct))\n"
)
DEMOTES = "    return ~np.isnan(fl) & (body < np.nan_to_num(fl, nan=0.0) - EPS) & (before != \"pinbar\")\n"

MUT = [
    # --- the floor function ---
    # the window includes the candle itself: test_the_window_is_strictly_before_the_candle (the 3rd pinbar's floor)
    ("u2_window_includes_candle", PF, LOOP,
     "        demoted[i] = bool(floor_demotes(body[i], btype[i], current))\n"
     "        if btype[i] == \"pinbar\" or (source == \"final\" and demoted[i]):\n"
     "            window.append(body[i])\n"
     "            if len(window) == size:\n"
     "                current = float(np.percentile(np.asarray(window), pct))\n"
     "        floor[i] = current\n"),
    # the window misses candle i - 1 (one candle late): test_the_window_is_strictly_before_the_candle (floor[3] NaN)
    ("u2_window_skips_prev", PF,
     "    current = np.nan                     # the floor of the next candle\n    for i in range(n):\n"
     "        floor[i] = current\n",
     "    current = np.nan\n    prev = np.nan\n    for i in range(n):\n        floor[i], prev = prev, current\n"),
    # "final" treated as "natural": test_natural_and_final_membership, test_the_seeds_demotions_count_in_the_final_window
    ("u2_final_as_natural", PF, "(source == \"final\" and demoted[i])", "(False and demoted[i])"),
    # nearest-rank instead of linear: test_the_percentile_is_numpy_linear (4.0, not 3.7)
    ("u2_nearest_rank", PF, "np.percentile(np.asarray(window), pct)",
     "np.percentile(np.asarray(window), pct, method=\"nearest\")"),
    # the window size hard-coded: every window-3 / 4 / 5 test (no full window → NaN)
    ("u2_window_hardcoded_90", PF, "    size = int(params.pinbar_floor_window)\n", "    size = 90\n"),
    # the percentile hard-coded: test_the_percentile_is_numpy_linear (p50 → 2.5, p90 gives 3.7)
    ("u2_percentile_hardcoded_90", PF, "    pct = float(params.pinbar_floor_percentile)\n", "    pct = 90.0\n"),
    # (review A1) the function's own predicate without EPS: test_the_function_uses_the_shared_predicate_with_eps
    # (its `demoted` on the 1e-15-under maru, exact arrays). First run SURVIVED: the pin then used a frame whose
    # float bodies landed the maru exactly AT its floor, so the EPS never mattered — re-pinned on exact arrays.
    ("u2_function_no_eps", PF, "        demoted[i] = bool(floor_demotes(body[i], btype[i], floor[i]))\n",
     "        demoted[i] = bool(body[i] < floor[i] and btype[i] != \"pinbar\")\n"),
    # (review A2) the window pre-filled with the seed's body-% pinbars only — the seed's demotions dropped from the
    # final window: test_the_seeds_demotions_count_in_the_final_window
    ("u2_seed_demotions_dropped", PF, "    sd = _body_pct_types(seed, params)\n",
     "    sd = _body_pct_types(seed, params)\n    sd = sd[sd[\"candle_type\"] == \"pinbar\"]\n"),
    # the seed-before-frame assert dropped: test_a_seed_not_strictly_before_the_frame_fails_loudly
    ("u2_no_seed_assert", PF, "    assert seed_last < frame_first, \\\n", "    assert True, \\\n"),
    # an unknown source not rejected: test_an_unknown_window_source_raises
    ("u2_source_not_checked", PF, "    if source not in WINDOW_SOURCES:\n", "    if False:\n"),
    # --- the shared predicate (step 4) ---
    # no EPS: test_pinbar_floor_boundary_eps_keeps_type, test_floor_demotes_is_the_step4_predicate (test_candles_v2)
    ("u2_demotes_no_eps", CV, DEMOTES,
     "    return ~np.isnan(fl) & (body < np.nan_to_num(fl, nan=0.0)) & (before != \"pinbar\")\n"),
    # the EPS sign flipped: the same pins
    ("u2_demotes_eps_sign", CV, DEMOTES,
     "    return ~np.isnan(fl) & (body < np.nan_to_num(fl, nan=0.0) + EPS) & (before != \"pinbar\")\n"),
    # a body-% pinbar flagged as demoted: test_pinbar_floor_flag_is_off_on_a_body_pct_pinbar + the predicate pin
    ("u2_demotes_any_pinbar", CV, DEMOTES,
     "    return ~np.isnan(fl) & (body < np.nan_to_num(fl, nan=0.0) - EPS)\n"),
    # --- the frame wiring ---
    # (review A1) the demotion-agreement assert dropped: test_compute_candle_features_asserts_the_demotions_agree
    ("u2_assert_dropped", CV, "        assert np.array_equal(out[\"is_floor_pinbar\"].to_numpy(dtype=bool), demoted), \\\n",
     "        assert True, \\\n"),
    # the body_floor seam ignored: test_compute_candle_features_takes_an_explicit_body_floor, the IP fixture pins
    ("u2_seam_ignored", CV, "    if body_floor is not None:\n        out = classify_candles(out, params, body_floor=body_floor)\n",
     "    if False:\n        out = classify_candles(out, params, body_floor=body_floor)\n"),
    # the frame's prior-maru pool fed by the seed: test_a_seed_fills_the_first_windows_and_only_the_windows
    ("u2_seed_into_maru_pool", CV, "    out = classify_big_flags(out, params, anchor_shifts=anchor_shifts)\n",
     "    if floor_seed is not None:\n"
     "        _s = classify_candles(compute_candle_metrics(floor_seed), params)\n"
     "        _all = classify_big_flags(pd.concat([_s, out], ignore_index=True), params, anchor_shifts=anchor_shifts)\n"
     "        out = _all.iloc[len(_s):].reset_index(drop=True)\n"
     "    else:\n"
     "        out = classify_big_flags(out, params, anchor_shifts=anchor_shifts)\n"),
    # apply_candle_classification drops the seed: test_apply_candle_classification_and_prepare_lower_tf_data_pass_the_seed
    ("u2_classifier_drops_seed", CC, "floor_seed=floor_seed, body_floor=body_floor)",
     "floor_seed=None, body_floor=body_floor)"),
    # prepare_lower_tf_data drops the seed: the same pin (its M15 half)
    ("u2_bridge_drops_seed", DB, "timeframe=timeframe, floor_seed=floor_seed)", "timeframe=timeframe)"),
    # run_pipeline: the H1 classification without its seed: test_run_pipeline_passes_the_h1_seed
    ("u2_pipeline_no_h1_seed", OR, "apply_candle_classification(df, floor_seed=floor_seeds.get(\"H1\"))",
     "apply_candle_classification(df, floor_seed=None)"),
    # run_pipeline → _run_multi_tf_dual without the M15 seed: test_run_pipeline_passes_the_m15_seed_to_the_m15_build
    ("u2_pipeline_no_m15_seed", OR, "            m15_floor_seed=floor_seeds.get(\"M15\"),\n",
     "            m15_floor_seed=None,\n"),
    # _run_multi_tf_dual → prepare_lower_tf_data without it: test_the_m15_build_passes_the_m15_seed
    ("u2_dual_no_m15_seed", OR, "prepare_lower_tf_data(m15_raw, floor_seed=m15_floor_seed)",
     "prepare_lower_tf_data(m15_raw)"),
    # --- the seed guard ---
    # the seed target 1 window: test_the_replay_fetches_both_seeds_for_two_windows ("(need 180)")
    ("u2_seed_one_window", RR, "    min_pinbars = int(p.pinbar_floor_seed_windows) * int(p.pinbar_floor_window)\n",
     "    min_pinbars = int(p.pinbar_floor_window)\n"),
    # the M15 seed fetched without M15: the same pin (the no-lower-TF call → {"H1"})
    ("u2_m15_seed_always", RR, "    if \"M15\" in lower_timeframes:\n", "    if True:\n"),
    # the guard's >= → >: test_the_guard_stops_at_exactly_enough
    ("u2_guard_strict_gt", RR, "        if n_pins >= int(min_pinbars):\n", "        if n_pins > int(min_pinbars):\n"),
    # each step fetches only its own 400 candles (no re-fetch of the whole span): test_the_guard_steps_back_…
    ("u2_step_not_cumulative", RR, "        start = (first - k * int(step_candles) * span).to_pydatetime()\n",
     "        start = (first - int(step_candles) * span).to_pydatetime()\n"),
    # (review A7) no `time < first_time` filter: test_the_seed_drops_the_candle_at_the_frames_first_time (the assert)
    ("u2_seed_no_time_filter", RR,
     "            raw = raw[pd.to_datetime(raw[\"time\"], utc=True) < first].reset_index(drop=True)\n",
     "            pass\n"),
    # … and without the assert after it: the same pin (401 candles, one AT the frame's first time)
    ("u2_seed_no_filter_no_assert", RR,
     "            raw = raw[pd.to_datetime(raw[\"time\"], utc=True) < first].reset_index(drop=True)\n"
     "        if raw.empty:\n            continue\n"
     "        assert pd.to_datetime(raw[\"time\"].iloc[-1], utc=True) < first\n",
     "            pass\n        if raw.empty:\n            continue\n"),
    # the seed's requests under the frame's prefix: test_a_failed_seed_request_retries_under_its_own_prefix
    ("u2_seed_prefix", RR, "prefix=\"[floor_seed]\", allow_empty=True)", "prefix=\"[auto_extend]\", allow_empty=True)"),
    # an empty step raises: test_an_empty_step_is_not_an_error
    ("u2_seed_empty_raises", RR, "prefix=\"[floor_seed]\", allow_empty=True)", "prefix=\"[floor_seed]\", allow_empty=False)"),
    # (review B12) the 5000-candle assert dropped: test_the_steps_stay_under_oandas_candle_limit
    ("u2_no_5000_assert", RR, "    assert step_candles * max_steps <= 5000,", "    assert True,"),
]
