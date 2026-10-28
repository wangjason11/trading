"""PLAN_W 9c-1 mutants (§3.6): the sub CTS KL zones returned + drawn hidden behind the `Sub CTS zones` button.

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py engine_v2/plans/plan_w_inputs/mutants_9c1.py

Killers: `tests/test_render_sub_projection.py::test_the_projection_returns_the_sub_cts_kl_zones_too`,
`tests/test_plan_w_sub_cts_zones.py` (the chart: exactly the CTS zones' elements tagged + hidden; the control cfg + the
HTML carry the group). The button's browser behaviour: the headless check (PLAN_W §3.7). LF anchors.
"""
EDM = "engine_v2/multitf/entity_df_mutation.py"
M15 = "engine_v2/charting/export_m15_chart.py"
PLT = "engine_v2/charting/export_plotly.py"

MUT = [
    ("v9c1_bos_only", EDM, '        source_kinds=("BOS", "CTS"),\n', ""),                      # the old projection
    ("v9c1_not_hidden", M15, "        tr.legendgroup = SUB_CTS_GROUP\n        tr.visible = False\n",
     "        tr.legendgroup = SUB_CTS_GROUP\n"),
    ("v9c1_shapes_untagged", M15, "        shape.legendgroup = SUB_CTS_GROUP\n        shape.visible = False\n",
     "        pass\n"),                                    # ("" would leave an empty loop body: a SyntaxError)
    ("v9c1_tags_bos_too", M15, '                if zone.source_kind == "CTS":\n', "                if True:\n"),
    ("v9c1_no_button_cfg", M15, "        sub_cts_group=SUB_CTS_GROUP))", "        ))"),
    ("v9c1_cfg_drops_group", PLT, '"subCtsGroup": sub_cts_group,', '"subCtsGroup": None,'),
]
