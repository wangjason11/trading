"""PLAN_U U1 mutants — run from the repo root:
    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \
        engine_v2/plans/qa_pinbar_floor_inputs/mutants_u1.py --out %TEMP%/mtU1

Each row: (id, relpath, old, new) — `old` occurs exactly once (text mode, LF anchors). The sketch after each = the
pin expected to kill it (written BEFORE the run). Not a mutant: dropping `~np.isnan(floor) &` in step 4 is EQUIVALENT
(a NaN floor becomes 0.0, and no body is below 0.0 - EPS).
"""
CV = "engine_v2/features/candles_v2.py"
IP = "engine_v2/patterns/institutional_pattern.py"
EP = "engine_v2/charting/export_plotly.py"
EM = "engine_v2/charting/export_m15_chart.py"

TINY = "    is_tiny_body = ~np.isnan(floor) & (body_len < np.nan_to_num(floor, nan=0.0) - EPS)\n"

MUT = [
    # the stack reads its FIRST candle's floor: test_a_stack_takes_the_floor_of_its_last_candle
    ("u1_stack_first", IP, "    floors = np.array([fr.floor[b] for _, b in multi], dtype=float)\n",
     "    floors = np.array([fr.floor[a] for a, _ in multi], dtype=float)\n"),
    # the stack gets no floor: test_stacks_use_the_frame_pip_floor (the 2.16-pip stack stays a maru)
    ("u1_stack_no_floor", IP, "fr.params, body_floor=floors)", "fr.params, body_floor=None)"),
    # no EPS guard: test_pinbar_floor_boundary_eps_keeps_type (a 3.0-pip body against the 3.0 floor)
    ("u1_no_eps", CV, TINY,
     "    is_tiny_body = ~np.isnan(floor) & (body_len < np.nan_to_num(floor, nan=0.0))\n"),
    # the EPS sign flipped: the same pin
    ("u1_eps_sign", CV, TINY,
     "    is_tiny_body = ~np.isnan(floor) & (body_len < np.nan_to_num(floor, nan=0.0) + EPS)\n"),
    # the flag set on a body-% pinbar too: test_pinbar_floor_flag_is_off_on_a_body_pct_pinbar
    ("u1_flag_any_pinbar", CV,
     '    out["is_floor_pinbar"] = is_tiny_body & (out["candle_type_before_floor"] != "pinbar").to_numpy()\n',
     '    out["is_floor_pinbar"] = is_tiny_body\n'),
    # the body-% type taken AFTER step 4: test_pinbar_floor_columns_on_a_demoted_maru (before = maru)
    ("u1_before_after_floor", CV, '    out.loc[is_tiny_body, "candle_type"] = "pinbar"\n',
     '    out.loc[is_tiny_body, "candle_type"] = "pinbar"\n    out["candle_type_before_floor"] = out["candle_type"]\n'),
    # the body-% type taken BEFORE the special-maru promotion (step 3): test_pinbar_floor_resets_special_maru
    ("u1_before_pre_special", CV, '    out["candle_type_before_floor"] = out["candle_type"]\n',
     '    out["candle_type_before_floor"] = np.where(out["is_special_maru"], "normal", out["candle_type"])\n'),
    # the floor column not written (NaN): test_pinbar_floor_columns_on_a_demoted_maru + the frame wiring pin
    ("u1_no_floor_column", CV, '    out["pinbar_body_floor"] = floor\n', '    out["pinbar_body_floor"] = np.nan\n'),
    # compute_candle_features passes no floor: test_compute_candle_features_applies_the_fixed_floor + the IP pins
    ("u1_features_no_floor", CV, "    out = classify_candles(out, params, body_floor=fixed_body_floor(out, params))\n",
     "    out = classify_candles(out, params, body_floor=None)\n"),
    # the pip ignores JPY: test_pinbar_floor_jpy_pip_size + test_jpy_stack_uses_the_frame_pair_pip
    ("u1_pip_no_jpy", CV, '    return 0.01 if "JPY" in str(df.attrs.get("pair", "")).upper() else 0.0001\n',
     "    return 0.0001\n"),
    # (review A3) a params floor without body_floor silently ignored: test_a_params_floor_without_body_floor_raises
    ("u1_no_params_raise", CV, "    if body_floor is None and float(params.pinbar_body_pip_floor) > 0.0:\n",
     "    if False:\n"),
    # (review A2) a disabled floor as 0.0 instead of None — the column 0.0, not NaN:
    # test_compute_candle_features_applies_the_fixed_floor (`off` all NaN)
    ("u1_disabled_as_zeros", CV, "    if floor_price <= EPS:\n        return None\n",
     "    if floor_price <= EPS:\n        return np.zeros(len(df))\n"),
    # the length check dropped: test_body_floor_length_must_match
    ("u1_no_length_check", CV, "    if len(floor) != len(out):\n", "    if False:\n"),
    # the hover's NA branch: test_candle_floor_hover_lines + the chart pins (candle 0)
    ("u1_hover_no_na", EP, "        if pd.isna(f):\n", "        if False:\n"),
    # the hover drops the body-% type on a floor pinbar: the same pins (candle 2)
    ("u1_hover_no_before", EP, "        elif b:\n", "        elif False:\n"),
    # the H1 template reads the wrong slot: test_the_candle_hover_shows_the_floor_line[H1-12]
    ("u1_h1_slot", EP, '        "candle_type=%{customdata[3]}<br>"\n        "%{customdata[12]}<br>"\n',
     '        "candle_type=%{customdata[3]}<br>"\n        "%{customdata[11]}<br>"\n'),
    # the M15 template reads the wrong slot: test_the_candle_hover_shows_the_floor_line[M15-14]
    ("u1_m15_slot", EM, '        "%{customdata[14]}<br>"\n', '        "%{customdata[13]}<br>"\n'),
]
