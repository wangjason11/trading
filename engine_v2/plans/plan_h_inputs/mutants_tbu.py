"""The IP tiebreaker update (PLAN_H §6 TBU-1 … TBU-5) — mutants for `mutant_runner.py` (full suite each).

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \
        engine_v2/plans/plan_h_inputs/mutants_tbu.py --jobs 9 --out <a short %TEMP% dir>

Every one must be KILLED.
"""
PZ = "engine_v2/zones/poi_zones.py"
ZR = "engine_v2/charting/_zone_render.py"
EZ = "engine_v2/debug/export_zones.py"
EDM = "engine_v2/multitf/entity_df_mutation.py"

MUT = [
    # the helper: the window, the clamp, the fill horizon, the direction, the structural assert, the returned index
    ("T01_window_starts_at_ic_plus_2", PZ,
     "        df, ic_idx + 1, hi, fib_state.cts_idx, config.fill_threshold,",
     "        df, ic_idx + 2, hi, fib_state.cts_idx, config.fill_threshold,"),
    ("T02_window_reaches_ic_plus_3", PZ,
     "    hi = min(ic_idx + config.ip_immediate_imbalance_candles, int(fib_state.cts_idx))",
     "    hi = min(ic_idx + config.ip_immediate_imbalance_candles + 1, int(fib_state.cts_idx))"),
    ("T03_window_only_ic_plus_1", PZ,
     "    hi = min(ic_idx + config.ip_immediate_imbalance_candles, int(fib_state.cts_idx))",
     "    hi = min(ic_idx + config.ip_immediate_imbalance_candles - 1, int(fib_state.cts_idx))"),
    ("T04_no_cts_clamp", PZ,
     "    hi = min(ic_idx + config.ip_immediate_imbalance_candles, int(fib_state.cts_idx))",
     "    hi = ic_idx + config.ip_immediate_imbalance_candles"),
    ("T05_fill_horizon_frame_end", PZ,
     "        df, ic_idx + 1, hi, fib_state.cts_idx, config.fill_threshold,",
     "        df, ic_idx + 1, hi, len(df) - 1, config.fill_threshold,"),
    ("T06_any_direction", PZ,
     "        direction=int(fib_state.struct_direction), evaluated_at=None,\n    ), None)",
     "        direction=None, evaluated_at=None,\n    ), None)"),
    ("T07_no_straddle_assert", PZ,
     "    assert inst.start_idx > ic_idx, (", "    assert True, ("),
    ("T08_returns_last_gap_candle", PZ,
     "    return int(inst.start_idx)", "    return int(inst.end_idx)"),
    ("T09_config_window_3", PZ,
     "    ip_immediate_imbalance_candles: int = 2", "    ip_immediate_imbalance_candles: int = 3"),
    # the helper's call site (every candidate)
    ("T10_helper_not_called", PZ,
     "immediate_imbalance_first_c2_idx=immediate_unfilled_imbalance_first_c2_idx(",
     "immediate_imbalance_first_c2_idx=None and immediate_unfilled_imbalance_first_c2_idx("),
    # the chain
    ("T11_sole_passer_not_final", PZ,
     "    if len(passing) == 1:", "    if False:"),
    ("T12_passers_ignored", PZ,
     "    in_play = passing or tied", "    in_play = tied"),
    ("T13_no_fallback_when_none_pass", PZ,
     "    in_play = passing or tied", "    in_play = passing"),
    ("T14_filter_over_all_patterns", PZ,
     "    passing = [r for r in tied if immediate[r.ic_idx] is not None]",
     "    passing = [r for r in patterns if immediate[r.ic_idx] is not None]"),
    ("T15_immediate_runner_up_oldest", PZ,
     "        runner_up = max(r.ic_idx for r in tied if r is not pick)",
     "        runner_up = min(r.ic_idx for r in tied if r is not pick)"),
    ("T16_immediate_count_passers", PZ,
     "(TIEBREAK_IMMEDIATE_UNFILLED_IMBALANCE, int(runner_up), len(tied),",
     "(TIEBREAK_IMMEDIATE_UNFILLED_IMBALANCE, int(runner_up), len(passing),"),
    ("T17_immediate_rule_mislabelled", PZ,
     "(TIEBREAK_IMMEDIATE_UNFILLED_IMBALANCE, int(runner_up), len(tied),",
     "(TIEBREAK_MOST_RECENT_BEFORE_IMBALANCE, int(runner_up), len(tied),"),
    ("T18_immediate_no_gap_field", PZ,
     "                                         int(immediate[pick.ic_idx])))",
     "                                         None))"),
    ("T19_type_count_tied", PZ,
     "(TIEBREAK_IC_CANDLE_TYPE, int(runner_up), len(in_play), None))",
     "(TIEBREAK_IC_CANDLE_TYPE, int(runner_up), len(tied), None))"),
    # the export + the mirror
    ("T20_record_field_dropped", PZ,
     "                immediate_imbalance_first_c2_idx=c.immediate_imbalance_first_c2_idx,\n", ""),
    ("T21_csv_column_empty", EZ,
     "            \"immediate_imbalance_first_c2_idx\": r.immediate_imbalance_first_c2_idx,",
     "            \"immediate_imbalance_first_c2_idx\": None,"),
    ("T22_mirror_not_shifted", EDM,
     "    \"ip_tb_imbalance_first_c2_idx\",\n    \"immediate_imbalance_first_c2_idx\",\n)",
     "    \"ip_tb_imbalance_first_c2_idx\",\n)"),
    # the hover
    ("T23_hover_immediate_branch", ZR,
     "    if tb == TIEBREAK_IMMEDIATE_UNFILLED_IMBALANCE:", "    if tb == \"immediate\":"),
    ("T24_hover_immediate_text", ZR,
     "                 f\"immediate unfilled imbalance {meta['ip_tb_imbalance_first_c2_idx']}\")",
     "                 f\"immediate unfilled imbalance {meta['ip_tb_runner_up_ic_idx']}\")"),
]
