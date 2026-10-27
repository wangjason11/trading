"""Mutants for "WVMI back on the chart" step 5 (2026-10-26, `user_spec_20261024_wvmi_on_chart.md`): the WVMI LP marks
above the volume bars — the pure helpers (`charting/_zone_render.wvmi_lp_*`), the plotly helpers
(`charting/export_plotly._add_wvmi_lp_traces` / `_volume_axis_range` / `_wvmi_lp_axis_layout`), the H1 chart's marks,
the M15 chart's `_render_wvmi_lp_marks`, the zoom JS's headroom on both charts and the style rows. Run from the repo root:

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \\
        engine_v2/plans/qa_chart_edits_inputs/mutants_wvmi_step5.py

`old` / `new` use LF (the runner reads in text mode). Every mutant must be KILLED by the full suite. The EQUIVALENT
ones are in `mutants_wvmi_step5_equivalent.py` (run them too — the PLAN_Q rule). Run 2026-10-26: 44 / 44 KILLED;
the equivalent one SURVIVES as claimed. Sketching these first found three test gaps (`relative_dir` / x read at the
as-of candle, an H1 as-of row, the zoomed slice) — the fixture was tightened before the run.
"""
ZR = "engine_v2/charting/_zone_render.py"
EP = "engine_v2/charting/export_plotly.py"
EM = "engine_v2/charting/export_m15_chart.py"
SR = "engine_v2/charting/style_registry.py"

_MARKS = "    return [(r, row) for r in records if not r.cycle_collapsed for row in r.lp_rows]\n"
_STATUS = "    return f\"{row.lp_status} ({row.lp_status_reason} at {int(row.lp_status_idx)})\"\n"
_AS_OF_IF = "    if int(row.as_of_idx) != int(row.lp_idx):\n"
_AS_OF = "        out.append(f\"as_of_idx{suffix}={int(row.as_of_idx)}\")\n"
_LAST_LINE = ("        f\"zone_side={record.zone_side} | breakout_momentum={_wvmi_momentum_text(record.breakout_momentum)} | \"\n"
              "        f\"pullback_momentum={_wvmi_momentum_text(row.pullback_momentum)}\",\n")
_ROW_Y = "            x=[p[0] for p in pts], y=[rows[side]] * len(pts),\n"
_STYLE = "            **_style(f\"wvmi.lp_dot.{kind}.{side}\"),\n"
_AXIS = "            mode=\"markers\", name=f\"{name_prefix} {side}\", showlegend=False, yaxis=\"y3\",\n"
_JS = "\"\"\".replace(\"__HEADROOM__\", repr(float(_style(\"volume.headroom\")[\"factor\"])))\n"
_Y3 = "    if wvmi_marks_drawn:\n        fig.update_layout(yaxis3=_wvmi_lp_axis_layout())\n"
_TOGGLE_EP = "    if volume_enabled and volume_cfg.get(\"wvmi_lps\", False):\n        wvmi_marks = []\n"
_TOGGLE_EM = ("    if volume_enabled and volume_cfg.get(\"wvmi_lps\", False):\n"
              "        wvmi_marks_drawn = _render_wvmi_lp_marks(")
_SUB_CALL = "    drawn = _add_wvmi_lp_traces(fig, sub_marks, kind=\"sub\", name_prefix=\"M15 WVMI LP\")\n"
_MAIN_CALL = "    drawn += _add_wvmi_lp_traces(fig, main_marks, kind=\"main\", name_prefix=\"H1 WVMI LP\")\n"
_LOH = "        m15_idx = _map_parent_idx_to_m15_hour_end(int(row.lp_idx), h1_df, m15_df)\n"
_MAIN_IDX = "            [*wvmi_lp_idx_fields(row, suffix=\"_1H\"), f\"lp_idx={m15_idx}\"], rec, row)))\n"
_REL = "        rel = _sub_relative_dir_at(row.lp_idx, sid_rec.relative_dir_segments or (), sid_rec.start_idx, sub_id)\n"
_SUB_T = "        t = time_by_idx.get(int(row.lp_idx))\n"
_SUB_ID_LINE = "            \"15M\", \"Sub\", f\"sub_id={sub_id} | cycle_id={rec.bos_cycle_id} | relative_dir={rel}\",\n"

MUT = [
    # the pure helpers: one mark per row, the collapsed filter, the hover fields
    ("marks_last_row_only", ZR, _MARKS,
     "    return [(r, row) for r in records if not r.cycle_collapsed for row in r.lp_rows[-1:]]\n"),
    ("marks_collapsed_shown", ZR, _MARKS, "    return [(r, row) for r in records for row in r.lp_rows]\n"),
    ("status_without_idx", ZR, _STATUS, "    return f\"{row.lp_status} ({row.lp_status_reason})\"\n"),
    ("as_of_always", ZR, _AS_OF_IF, "    if True:\n"),
    ("as_of_never", ZR, _AS_OF_IF, "    if False:\n"),
    ("as_of_without_suffix", ZR, _AS_OF, "        out.append(f\"as_of_idx={int(row.as_of_idx)}\")\n"),
    ("momentum_2dp", ZR, "    return \"NA (not computed)\" if value is None else f\"{float(value):.3f}\"\n",
     "    return \"NA (not computed)\" if value is None else f\"{float(value):.2f}\"\n"),
    ("momentum_na_dash", ZR, "    return \"NA (not computed)\" if value is None else f\"{float(value):.3f}\"\n",
     "    return \"-\" if value is None else f\"{float(value):.3f}\"\n"),
    ("pullback_from_record", ZR, _LAST_LINE, _LAST_LINE.replace("row.pullback_momentum", "record.pullback_momentum")),
    ("zone_side_named_wvmi_dir", ZR, _LAST_LINE, _LAST_LINE.replace("zone_side=", "wvmi_dir=")),
    ("owner_line_unbolded", ZR, "        f\"TF={tf}\", f\"<b>{owner} WVMI</b>\", id_line,\n",
     "        f\"TF={tf}\", f\"{owner} WVMI\", id_line,\n"),
    # the plotly helpers: side -> row / colour, the axis, the headroom
    ("side_rows_swapped", EP, _ROW_Y,
     "            x=[p[0] for p in pts], y=[rows[\"buy\" if side == \"sell\" else \"sell\"]] * len(pts),\n"),
    ("side_colour_swapped", EP, _STYLE,
     "            **_style(f\"wvmi.lp_dot.{kind}.{'buy' if side == 'sell' else 'sell'}\"),\n"),
    ("marks_on_volume_axis", EP, _AXIS, _AXIS.replace("yaxis=\"y3\"", "yaxis=\"y2\"")),
    ("side_guard_dropped", EP,
     "    assert all(m[1] in rows for m in marks), f\"[chart] WVMI LP zone_side not in {sorted(rows)}\"\n", ""),
    ("headroom_dropped", EP, "    return [0, vmax * float(_style(\"volume.headroom\")[\"factor\"])]\n",
     "    return [0, vmax]\n"),
    ("headroom_zero_volume_kept", EP, "    if not vmax > 0:\n        return None\n", ""),
    ("y3_not_over_volume", EP, "    return dict(overlaying=\"y2\", anchor=\"x\", range=[0, 1], fixedrange=True, visible=False,\n",
     "    return dict(overlaying=\"y\", anchor=\"x\", range=[0, 1], fixedrange=True, visible=False,\n"),
    ("y3_zoomable", EP, "    return dict(overlaying=\"y2\", anchor=\"x\", range=[0, 1], fixedrange=True, visible=False,\n",
     "    return dict(overlaying=\"y2\", anchor=\"x\", range=[0, 1], fixedrange=False, visible=False,\n"),
    # the H1 chart
    ("h1_toggle_ignored", EP, _TOGGLE_EP, "    if volume_enabled:\n        wvmi_marks = []\n"),
    ("h1_x_at_as_of", EP, "                dfx.loc[row.lp_idx, COL_TIME], rec.zone_side,\n",
     "                dfx.loc[row.as_of_idx, COL_TIME], rec.zone_side,\n"),
    ("h1_slice_unchecked", EP, "            if row.lp_idx not in dfx.index:\n                continue      # a zoomed export's slice\n",
     ""),
    ("h1_drawn_as_dots", EP, "        wvmi_marks_drawn = _add_wvmi_lp_traces(fig, wvmi_marks, kind=\"main\", name_prefix=\"WVMI LP\")\n",
     "        wvmi_marks_drawn = _add_wvmi_lp_traces(fig, wvmi_marks, kind=\"sub\", name_prefix=\"WVMI LP\")\n"),
    ("h1_tf_label", EP, "                wvmi_lp_hovertext(\"1H\", \"Main\", f\"sid={rec.bos_structure_id} | cycle_id={rec.bos_cycle_id}\",\n",
     "                wvmi_lp_hovertext(\"15M\", \"Main\", f\"sid={rec.bos_structure_id} | cycle_id={rec.bos_cycle_id}\",\n"),
    ("h1_y3_layout_dropped", EP, _Y3, ""),
    ("h1_js_headroom_unreplaced", EP, _JS, "\"\"\"\n"),
    ("h1_volume_range_dropped", EP, "                range=_volume_axis_range(dfx[COL_V].astype(float)),  # headroom for the WVMI LP rows\n", ""),
    # the M15 chart
    ("m15_toggle_ignored", EM, _TOGGLE_EM, "    if volume_enabled:\n        wvmi_marks_drawn = _render_wvmi_lp_marks("),
    ("m15_h1_at_hour_start", EM, _LOH, "        m15_idx = int(row.lp_idx) * 4\n"),
    ("m15_h1_without_m15_idx", EM, _MAIN_IDX, "            [*wvmi_lp_idx_fields(row, suffix=\"_1H\")], rec, row)))\n"),
    ("m15_h1_without_suffix", EM, _MAIN_IDX, "            [*wvmi_lp_idx_fields(row), f\"lp_idx={m15_idx}\"], rec, row)))\n"),
    ("m15_h1_tf_label", EM, "            \"1H>>15M\", \"Main\", f\"sid={rec.bos_structure_id} | cycle_id={rec.bos_cycle_id}\",\n",
     "            \"1H\", \"Main\", f\"sid={rec.bos_structure_id} | cycle_id={rec.bos_cycle_id}\",\n"),
    ("m15_sub_drawn_as_circles", EM, _SUB_CALL, _SUB_CALL.replace("kind=\"sub\"", "kind=\"main\"")),
    ("m15_h1_drawn_as_dots", EM, _MAIN_CALL, _MAIN_CALL.replace("kind=\"main\"", "kind=\"sub\"")),
    ("m15_circles_under_dots", EM, _SUB_CALL + _MAIN_CALL,
     "    drawn = _add_wvmi_lp_traces(fig, main_marks, kind=\"main\", name_prefix=\"H1 WVMI LP\")\n"
     "    drawn += _add_wvmi_lp_traces(fig, sub_marks, kind=\"sub\", name_prefix=\"M15 WVMI LP\")\n"),
    ("m15_rel_at_as_of", EM, _REL, _REL.replace("(row.lp_idx,", "(row.as_of_idx,")),
    ("m15_rel_wrong_sub", EM, "        sid_rec = sid_by_sub_id[sub_id]\n", "        sid_rec = sid_records[0]\n"),
    ("m15_sub_x_at_as_of", EM, _SUB_T, "        t = time_by_idx.get(int(row.as_of_idx))\n"),
    ("m15_sub_cycle_is_sid", EM, _SUB_ID_LINE, _SUB_ID_LINE.replace("rec.bos_cycle_id", "rec.bos_structure_id")),
    ("m15_sub_records_from_h1", EM, "    for rec, row in wvmi_lp_marks(m15_df.attrs.get(\"wvmi\", [])):\n",
     "    for rec, row in wvmi_lp_marks(h1_df.attrs.get(\"wvmi\", [])):\n"),
    ("m15_y3_layout_dropped", EM, _Y3, ""),
    ("m15_js_headroom_unreplaced", EM, _JS, "\"\"\"\n"),
    ("m15_volume_range_dropped", EM, "                        range=_volume_axis_range(dfx[COL_V].astype(float))),   # headroom for the WVMI LP rows\n",
     "                        ),\n"),
    # the style rows: buy one step above sell (the user, round 2: "The green dot should be slightly above the red one")
    ("rows_swapped", SR, "        \"sell\": 0.80,\n        \"buy\": 0.91,\n", "        \"sell\": 0.91,\n        \"buy\": 0.80,\n"),
]
