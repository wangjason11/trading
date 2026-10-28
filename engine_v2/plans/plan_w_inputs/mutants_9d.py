"""PLAN_W §4 mutants: the KL zone hover's `kind=` line (`charting/_zone_render.py`).

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py engine_v2/plans/plan_w_inputs/mutants_9d.py

Killers (`tests/test_zone_hover.py`): `test_main_kl_hover_renders_the_spec_order`,
`test_sub_kl_hover_carries_the_sub_identity_and_the_first_record`, `test_the_kl_hover_names_the_zone_kind_on_main_and_
sub_zones`. LF anchors.
"""
ZR = "engine_v2/charting/_zone_render.py"
MUT = [
    ("v9d_no_line", ZR, '        "kind=%{customdata[10]}<br>"\n', ""),
    ("v9d_wrong_slot", ZR, '        "kind=%{customdata[10]}<br>"\n', '        "kind=%{customdata[9]}<br>"\n'),
    ("v9d_constant_kind", ZR, "            str(zone.source_kind)]", '            "BOS"]'),
    ("v9d_no_guard", ZR, '    assert zone.source_kind in ("BOS", "CTS"), f"[chart] KL zone source_kind {zone.source_kind!r}"\n', ""),
]
