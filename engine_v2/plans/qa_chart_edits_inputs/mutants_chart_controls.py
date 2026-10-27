"""Mutants for item 5's final part, the chart control bar (2026-10-27, `user_spec_20261027_parent_toggle_and_zoom.md`):
the parent-overlay group (`export_m15_chart._tag_parent_overlay` + its call around `_render_h1_overlay`), the bar's
config + injection (`export_plotly._chart_controls_cfg` / `_inject_chart_controls_js`), both exporters' calls and the
`chart.controls` knobs. Run from the repo root:

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \\
        engine_v2/plans/qa_chart_edits_inputs/mutants_chart_controls.py

`old` / `new` use LF (the runner reads in text mode). Every mutant must be KILLED by the full suite; each was sketched
against `tests/test_chart_controls.py` before the run (the killing assertion is named per group). The script's
behaviour in a browser (chart_controls.js) is not mutated here — it is checked headless (`proto_headless_check.py`).
"""
EM = "engine_v2/charting/export_m15_chart.py"
EP = "engine_v2/charting/export_plotly.py"
SR = "engine_v2/charting/style_registry.py"

_MEASURE = "    first_trace, first_shape = len(fig.data), len(fig.layout.shapes)\n"
_TAG = "    _tag_parent_overlay(fig, first_trace, first_shape)\n"
_OVERLAY = ("    _render_h1_overlay(fig, dfx, h1_df, h1_to_m15, m15_to_h1,\n"
            "                       state_cfg, struct_cfg, zone_cfg)\n")
_TRIG = "    _render_proximity_triggers_overlay(fig, dfx, h1_df, wick_offset)\n"
_TAG_TO_TRIG = (_TAG + "\n    # Parent (H1) zone-proximity triggers, anchored at the matching M15\n"
                "    # sub-candle. Parent-event overlay, so no §16.5 owner filter applies.\n" + _TRIG)
_M15_INJ = ("    _inject_chart_controls_js(html_path, _chart_controls_cfg(\n"
            "        dfx, tf=\"15M\", idx_1h=idx_1h_col, parent_tf=\"1H\", parent_group=PARENT_OVERLAY_GROUP))\n")
_M15_VOL = "    if volume_enabled:\n        _inject_volume_autoscale_js(html_path)\n"
_H1_INJ = "    _inject_chart_controls_js(html_path, _chart_controls_cfg(dfx, tf=\"1H\"))\n"
_TIMES = "    times = pd.to_datetime(dfx[COL_TIME], utc=True)\n"
_ORDER = "    assert times.is_monotonic_increasing, \"[chart] the control bar's candle table needs the frame in time order\"\n"
_ROWS = "               for i, t, lo, hi in zip(dfx.index, times.dt.strftime(\"%Y-%m-%d %H:%M:%S\"), dfx[COL_L], dfx[COL_H])]\n"
_ALIGN = "        assert len(idx_1h) == len(candles), \"[chart] idx_1h must align with the frame\"\n"
_IDX1H = "        candles = [row + [None if h in (\"\", None) else int(h)] for row, h in zip(candles, idx_1h)]\n"

MUT = [
    # --- the group: test_the_group_is_exactly_what_the_h1_overlay_draws (+ the real-overlay pin) ---
    ("no_tagging", EM, _TAG, ""),
    ("trace_slice_plus1", EM, "    for tr in fig.data[first_trace:]:\n", "    for tr in fig.data[first_trace + 1:]:\n"),
    ("trace_slice_minus1", EM, "    for tr in fig.data[first_trace:]:\n", "    for tr in fig.data[first_trace - 1:]:\n"),
    ("shape_slice_plus1", EM, "    for shape in fig.layout.shapes[first_shape:]:\n",
     "    for shape in fig.layout.shapes[first_shape + 1:]:\n"),
    ("shape_slice_minus1", EM, "    for shape in fig.layout.shapes[first_shape:]:\n",
     "    for shape in fig.layout.shapes[first_shape - 1:]:\n"),
    ("shapes_untagged", EM, "        shape.legendgroup = PARENT_OVERLAY_GROUP\n", "        pass\n"),
    ("tag_after_triggers", EM, _TAG_TO_TRIG, _TAG_TO_TRIG.replace(_TAG, "") + _TAG),
    ("measure_after_overlay", EM, _MEASURE + _OVERLAY, _OVERLAY + _MEASURE),
    ("group_name", EM, "PARENT_OVERLAY_GROUP = \"parent_overlay\"\n", "PARENT_OVERLAY_GROUP = \"parent\"\n"),
    # --- the M15 call: test_m15_chart_carries_the_bar_with_its_parent_and_the_hover_idx_1h / ..._no_volume_panel ---
    ("m15_no_parent_group", EM, "parent_group=PARENT_OVERLAY_GROUP))\n", "parent_group=None))\n"),
    ("m15_no_parent_tf", EM, "parent_tf=\"1H\", parent_group", "parent_tf=None, parent_group"),
    ("m15_no_idx_1h", EM, "idx_1h=idx_1h_col,", "idx_1h=None,"),
    ("m15_tf", EM, "        dfx, tf=\"15M\", idx_1h", "        dfx, tf=\"1H\", idx_1h"),
    ("m15_not_injected", EM, _M15_INJ, ""),
    ("m15_before_volume_js", EM, _M15_VOL + _M15_INJ, _M15_INJ + _M15_VOL),
    ("m15_only_with_volume", EM, _M15_VOL + _M15_INJ,
     _M15_VOL + "".join("    " + line + "\n" for line in _M15_INJ.splitlines())),
    # --- the H1 call: test_h1_chart_carries_the_bar_without_a_parent / ..._no_volume_panel[h1] ---
    ("h1_not_injected", EP, _H1_INJ, ""),
    ("h1_tf", EP, _H1_INJ, _H1_INJ.replace("tf=\"1H\"", "tf=\"15M\"")),
    ("h1_only_with_volume", EP, _H1_INJ, "    " + _H1_INJ),
    # --- the config: test_cfg_candle_table_and_knobs / test_cfg_time_is_the_utc_wall_clock / test_cfg_refuses_... ---
    ("time_not_utc", EP, _TIMES, "    times = pd.to_datetime(dfx[COL_TIME])\n"),
    ("time_format", EP, _ROWS, _ROWS.replace("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")),
    ("low_high_swapped", EP, _ROWS, _ROWS.replace("dfx[COL_L], dfx[COL_H]", "dfx[COL_H], dfx[COL_L]")),
    ("unmapped_idx_1h", EP, _IDX1H, _IDX1H.replace("None if h in (\"\", None) else int(h)", "-1 if h == \"\" else int(h)")),
    ("no_order_assert", EP, _ORDER, ""),
    ("no_align_assert", EP, _ALIGN, ""),
    ("half_knob", EP, "\"half\": int(st[\"idx_half_window\"])", "\"half\": int(st[\"min_candles\"])"),
    ("pad_knob", EP, "\"pricePad\": float(st[\"price_pad\"])", "\"pricePad\": float(st[\"wheel_step\"])"),
    # --- the injection: test_inject_appends_the_script_with_its_cfg ---
    ("cfg_not_filled", EP, "js.replace(\"__CFG__\", json.dumps(cfg))", "js.replace(\"__CFG__\", json.dumps({}))"),
    # --- the knobs: test_cfg_candle_table_and_knobs (the values the user looked at) ---
    ("style_half", SR, "        \"idx_half_window\": 48,", "        \"idx_half_window\": 24,"),
    ("style_pad", SR, "        \"price_pad\": 0.06,", "        \"price_pad\": 0.05,"),
    ("style_min", SR, "        \"min_candles\": 10,", "        \"min_candles\": 12,"),
    ("style_wheel", SR, "        \"wheel_step\": 1.2,", "        \"wheel_step\": 1.25,"),
    ("style_mark", SR, "\"idx_mark_color\": \"rgba(30,120,255,0.18)\"", "\"idx_mark_color\": \"rgba(30,120,255,0.3)\""),
]
