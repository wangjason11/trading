"""PLAN_V 8c mutants (section 3): the qualified-candle rule and the three wave-candle renderers' vol_dir colour.

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py engine_v2/plans/plan_v_inputs/mutants_8c.py

Killers: tests/test_institutional_pattern.py::test_qualified_rule_is_the_agreed_direction_incl_the_volume_only_doji,
tests/test_wave_candles.py::TestIsQualified, tests/test_wvmi.py::...::test_a_volume_only_doji_is_a_candidate (the rule);
tests/test_plan_v_registry_styles.py (the renderers: two rendered, the M15 sub site by a source count).
"""
WC = "engine_v2/zones/wave_candles.py"
M15 = "engine_v2/charting/export_m15_chart.py"
H1 = "engine_v2/charting/export_plotly.py"

MUT = [
    # the old rule: the candle direction must match as well
    ("v8c_rule_requires_direction", WC,
     '    direction = int(df.loc[idx, "direction"])\n    vol_dir = int(df.loc[idx, "vol_dir"]) if "vol_dir" in df.columns else 0\n',
     '    direction = int(df.loc[idx, "direction"])\n    if direction != required_dir:\n        return False\n'
     '    vol_dir = int(df.loc[idx, "vol_dir"]) if "vol_dir" in df.columns else 0\n'),
    # each renderer back to skipping a direction-0 wave candle
    ("v8c_m15_sub_no_vol_dir_colour", M15,
     'candle_dir = int(lt_df.iloc[idx]["vol_dir"]) if "vol_dir" in lt_df.columns else 0',
     'candle_dir = 0'),
    ("v8c_h1ov_no_vol_dir_colour", M15,
     'candle_dir = int(h1_df.loc[idx, "vol_dir"]) if "vol_dir" in h1_df.columns else 0',
     'candle_dir = 0'),
    ("v8c_h1_no_vol_dir_colour", H1,
     'candle_dir = int(dfx.loc[idx, "vol_dir"]) if "vol_dir" in dfx.columns else 0',
     'candle_dir = 0'),
]
