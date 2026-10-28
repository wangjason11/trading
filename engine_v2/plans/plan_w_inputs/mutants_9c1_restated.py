"""PLAN_W 9c-1 — the control bar's mutants (`qa_chart_edits_inputs/mutants_chart_controls.py`) whose anchors 9c-1 made
stale (`anchor_scan_9c1.txt`), re-anchored on the new code, each mutation the original's:
  - the four slice mutants of `_tag_parent_overlay`: its loop lines now also occur in `_tag_sub_cts_zone` — the anchor
    takes the next line (`… = PARENT_OVERLAY_GROUP`) to stay unique;
  - the four on the M15 chart's control-config call: the call gained `sub_cts_group=SUB_CTS_GROUP`.

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py engine_v2/plans/plan_w_inputs/mutants_9c1_restated.py
"""
M15 = "engine_v2/charting/export_m15_chart.py"
_TR = "    for tr in fig.data[first_trace:]:\n        tr.legendgroup = PARENT_OVERLAY_GROUP\n"
_SH = "    for shape in fig.layout.shapes[first_shape:]:\n        shape.legendgroup = PARENT_OVERLAY_GROUP\n"
_CALL = ('    _inject_chart_controls_js(html_path, _chart_controls_cfg(\n'
         '        dfx, tf="15M", idx_1h=idx_1h_col, parent_tf="1H", parent_group=PARENT_OVERLAY_GROUP,\n'
         '        sub_cts_group=SUB_CTS_GROUP))\n')
_VOL = "    if volume_enabled:\n        _inject_volume_autoscale_js(html_path)\n"

MUT = [
    ("trace_slice_plus1_9c1", M15, _TR, _TR.replace("[first_trace:]", "[first_trace + 1:]")),
    ("trace_slice_minus1_9c1", M15, _TR, _TR.replace("[first_trace:]", "[first_trace - 1:]")),
    ("shape_slice_plus1_9c1", M15, _SH, _SH.replace("[first_shape:]", "[first_shape + 1:]")),
    ("shape_slice_minus1_9c1", M15, _SH, _SH.replace("[first_shape:]", "[first_shape - 1:]")),
    ("m15_no_parent_group_9c1", M15, "parent_group=PARENT_OVERLAY_GROUP,\n        sub_cts_group=SUB_CTS_GROUP))\n",
     "parent_group=None,\n        sub_cts_group=SUB_CTS_GROUP))\n"),
    ("m15_not_injected_9c1", M15, _CALL, ""),
    ("m15_before_volume_js_9c1", M15, _VOL + _CALL, _CALL + _VOL),
    ("m15_only_with_volume_9c1", M15, _VOL + _CALL,
     _VOL + "".join("    " + line + "\n" for line in _CALL.splitlines())),
]
