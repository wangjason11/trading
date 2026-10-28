"""PLAN_V 8a mutants (§1.4): each registry read reverted to the literal it replaced + the H1 assert removed.

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py engine_v2/plans/plan_v_inputs/mutants_8a.py

Every mutant is byte-equal on the replay (the registry holds the literal's value); its killer patches the key to a
sentinel and shows the drawn element does not follow (`tests/test_plan_v_registry_styles.py`). Anchors use LF (the
runner reads text mode).
"""
M15 = "engine_v2/charting/export_m15_chart.py"
H1 = "engine_v2/charting/export_plotly.py"
RR = "engine_v2/run_replay.py"

_OUT_NEW = ('                sub_outline_color = _rgba_from_rgb(str(outline_st.get("rgb", "0, 0, 0")), line_op)\n'
            '                sub_outline_w = float(outline_st.get("width", 0.5))')
_OUT_OLD = ('                sub_outline_color = f"rgba(0, 0, 0, {line_op})"\n'
            '                sub_outline_w = 0.5')
_HB6 = '{"width": 6, "color": "rgba(0,0,0,0)"}'

MUT = [
    ("v8a_kl_outline_literal", M15,
     "tracks the line opacity for tier-consistent dimming.\n" + '                outline_st = _style("zone.m15.outline")\n'
     + _OUT_NEW,
     "tracks the line opacity for tier-consistent dimming.\n" + '                outline_st = _style("zone.m15.outline")\n'
     + _OUT_OLD),
    ("v8a_poi_outline_literal", M15,
     "thin black 0.5px, tier-faded).\n" + '                outline_st = _style("zone.m15.outline")\n' + _OUT_NEW,
     "thin black 0.5px, tier-faded).\n" + '                outline_st = _style("zone.m15.outline")\n' + _OUT_OLD),
    ("v8a_m15_kl_hitbox_literal", M15,
     'kl_hover_line = _style("zone.kl.hover_line").get("line", ' + _HB6 + ")",
     "kl_hover_line = " + _HB6),
    ("v8a_m15_poi_hitbox_literal", M15,
     'poi_hover_line = _style("zone.poi.hover_line").get("line", ' + _HB6 + ")",
     "poi_hover_line = " + _HB6),
    ("v8a_h1ov_kl_hitbox_literal", M15,
     'h1_hover_line = _style("zone.kl.hover_line").get("line", ' + _HB6 + ")",
     "h1_hover_line = " + _HB6),
    ("v8a_h1ov_poi_hitbox_literal", M15,
     'h1_hover_line = _style("zone.poi.hover_line").get("line", ' + _HB6 + ")",
     "h1_hover_line = " + _HB6),
    ("v8a_m15_sub_wave_hitbox_literal", M15,
     '\n                        line=_style("wave_candle.hover_line").get("line", dict(width=8, color="rgba(0,0,0,0)")),',
     '\n                        line=dict(width=8, color="rgba(0,0,0,0)"),'),
    ("v8a_h1ov_wave_hitbox_literal", M15,
     '\n                    mode="lines", showlegend=False,\n'
     '                    line=_style("wave_candle.hover_line").get("line", dict(width=8, color="rgba(0,0,0,0)")),',
     '\n                    mode="lines", showlegend=False,\n'
     '                    line=dict(width=8, color="rgba(0,0,0,0)"),'),
    ("v8a_h1_wave_hitbox_literal", H1,
     'line=_style("wave_candle.hover_line").get("line", dict(width=8, color="rgba(0,0,0,0)")),',
     'line=dict(width=8, color="rgba(0,0,0,0)"),'),
    ("v8a_m15_fib_hitbox_literal", M15,
     'line=_style("fib.hover_line").get("line", dict(width=6, color="rgba(0,0,0,0)")),',
     'line=dict(width=6, color="rgba(0,0,0,0)"),'),
    ("v8a_h1_fib_hitbox_literal", H1,
     'line=_style("fib.hover_line").get("line", dict(width=6, color="rgba(0,0,0,0)")),',
     'line=dict(width=6, color="rgba(0,0,0,0)"),'),
    ("v8a_prev_bos_label_literal", M15,
     'showlegend=False,\n                    hoverlabel=dict(_style("hoverlabel.m15_structure")),',
     'showlegend=False,\n                    hoverlabel=dict(bgcolor="royalblue", font_color="white"),'),
    ("v8a_dot_label_literal", M15,
     'customdata=cd,\n        hoverlabel=dict(_style("hoverlabel.m15_structure")),',
     'customdata=cd,\n        hoverlabel=dict(bgcolor="royalblue", font_color="white"),'),
    ("v8a_proximity_label_literal", M15,
     'hoverlabel=dict(_style("hoverlabel.zone_proximity")),',
     'hoverlabel=dict(bgcolor="black", font_color="white"),'),
    ("v8a_h1_assert_removed", RR,
     '    assert CONFIG.timeframe == "H1", f"[run_replay] the pipeline is H1-main; config.timeframe={CONFIG.timeframe!r}"',
     '    pass'),
]
