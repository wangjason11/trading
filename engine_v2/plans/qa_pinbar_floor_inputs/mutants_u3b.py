"""PLAN_U U3b mutants (§6b: the seed guard steps back until final mode converges) — run from the repo root:
    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \
        engine_v2/plans/qa_pinbar_floor_inputs/mutants_u3b.py --out %TEMP%/mtU3b

Each row: (id, relpath, old, new) — `old` occurs exactly once (text mode, LF anchors). The sketch after each = the
pin expected to kill it (written BEFORE the run; the tests: `test_plan_u_rolling_floor.py`). Not mutants
(equivalent): `len(pin_rows) < size` -> `<=` in `floor_seed_converged` (exactly `size` pinbars leave the cut seed
none, so its window differs anyway); `fetch_floor_seeds` not passing `params=p` (the default is `CandleParams()`).
"""
PF = "engine_v2/features/pinbar_body_floor.py"
RR = "engine_v2/run_replay.py"

GUARD_IF = ("        if n_pins >= int(min_pinbars) and floor_seed_converged(typed[\"body_len\"], typed[\"candle_type\"], "
            "params):\n")

MUT = [
    # --- the predicate (`features/pinbar_body_floor.floor_seed_converged`) ---
    # always converged: test_the_seed_converges_when_both_runs_end_on_the_same_window (the first False),
    # test_the_guard_steps_on_until_the_final_window_converges (a stop at step 1)
    ("u3b_always_converged", PF, "    if len(pin_rows) < size:\n        return False\n", "    return True\n"),
    # the cut AT the window-th pinbar (one early): test_the_cut_is_just_after_the_window_th_pinbar
    ("u3b_cut_at_the_pinbar", PF, "    cut = int(pin_rows[size - 1]) + 1\n", "    cut = int(pin_rows[size - 1])\n"),
    # the cut after the (window+1)-th pinbar (one late): test_natural_mode_converges_once_the_cut_seed_holds_a_window
    # (4 pinbars: the cut seed keeps 1 -> False)
    ("u3b_cut_one_late", PF, "    cut = int(pin_rows[size - 1]) + 1\n", "    cut = int(pin_rows[size]) + 1\n"),
    # the next floors compared instead of the windows: test_converged_compares_the_window_not_the_floor
    ("u3b_compares_floors", PF, "    return full == tuple(r + cut for r in part)\n",
     "    pct = float(params.pinbar_floor_percentile)\n"
     "    return len(part) == size and bool(np.percentile(body[list(full)], pct) == "
     "np.percentile(body[cut:][list(part)], pct))\n"),
    # the cut run's rows not shifted back by the cut: test_the_seed_converges_when_both_runs_end_on_the_same_window
    # (the True), test_natural_mode_converges_once_the_cut_seed_holds_a_window (the Trues)
    ("u3b_rows_not_shifted", PF, "    return full == tuple(r + cut for r in part)\n", "    return full == tuple(part)\n"),
    # no short check (fewer than a window -> IndexError): test_natural_mode_converges_… (the single-pinbar seed)
    ("u3b_no_short_check", PF, "    if len(pin_rows) < size:\n        return False\n", ""),
    # --- the guard (`run_replay.fetch_floor_seed`) ---
    # the guard ignores convergence: test_the_guard_steps_on_until_the_final_window_converges (1 call, not 2),
    # test_the_guard_raises_when_the_window_never_converges (returns at step 1)
    ("u3b_guard_ignores_convergence", RR, GUARD_IF, "        if n_pins >= int(min_pinbars):\n"),
    # the count OR convergence: test_the_guard_steps_back_until_the_seed_holds_enough_pinbars (step 1 converged with
    # 200 < 300 pinbars would stop)
    ("u3b_guard_count_or_converged", RR, GUARD_IF, GUARD_IF.replace(") and floor_seed", ") or floor_seed")),
    # >= -> > on the count: test_the_guard_stops_at_exactly_enough (U2's pin, now with the convergence),
    # test_the_guard_in_natural_mode_stops_at_the_count (4 of 4)
    ("u3b_guard_count_strict_gt", RR, GUARD_IF, GUARD_IF.replace("n_pins >= int", "n_pins > int")),
    # the check on the default settings, not `params`: test_the_guard_steps_on_until_the_final_window_converges (a
    # 90-window check never converges on 10 candles -> raises)
    ("u3b_guard_default_params", RR, GUARD_IF, GUARD_IF.replace("], params):", "], CandleParams()):")),
    # a not-converged seed reported as a short count: test_the_guard_raises_when_the_window_never_converges
    ("u3b_error_says_count", RR, "    if n_pins < int(min_pinbars):\n", "    if True:\n"),
    # --- U2's loop mutants re-anchored on the U3b loop (its window holds rows; `mutants_u2.py`'s anchors for these no
    # longer match since U3b; its `u2_guard_strict_gt` = `u3b_guard_count_strict_gt` above). Same killers as at U2.
    # the window includes the candle itself: test_the_window_is_strictly_before_the_candle (the 3rd pinbar's floor)
    ("u2r_window_includes_candle", PF, LOOP_U3B := (
        "        floor[i] = current\n"
        "        demoted[i] = bool(floor_demotes(body[i], btype[i], floor[i]))\n"
        "        if btype[i] == \"pinbar\" or (source == \"final\" and demoted[i]):\n"
        "            window.append(i)\n"
        "            if len(window) == size:\n"
        "                current = float(np.percentile(body[list(window)], pct))\n"),
     "        demoted[i] = bool(floor_demotes(body[i], btype[i], current))\n"
     "        if btype[i] == \"pinbar\" or (source == \"final\" and demoted[i]):\n"
     "            window.append(i)\n"
     "            if len(window) == size:\n"
     "                current = float(np.percentile(body[list(window)], pct))\n"
     "        floor[i] = current\n"),
    # nearest-rank instead of linear: test_the_percentile_is_numpy_linear (4.0, not 3.7)
    ("u2r_nearest_rank", PF, "np.percentile(body[list(window)], pct)",
     "np.percentile(body[list(window)], pct, method=\"nearest\")"),
    # the loop's window size hard-coded: every window-3 / 4 / 5 test (no full window -> NaN)
    ("u2r_window_hardcoded_90", PF, "    size = int(params.pinbar_floor_window)\n    pct = float(params.pinbar_floor_percentile)\n",
     "    size = 90\n    pct = float(params.pinbar_floor_percentile)\n"),
    # the predicate's window size hard-coded (the cut after the 90th pinbar): the window-2 predicate tests (True cases)
    ("u3b_predicate_window_hardcoded_90", PF,
     "    size = int(params.pinbar_floor_window)\n    pin_rows = np.flatnonzero(btype == \"pinbar\")\n",
     "    size = 90\n    pin_rows = np.flatnonzero(btype == \"pinbar\")\n"),
]
