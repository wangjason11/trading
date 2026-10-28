"""PLAN_W 9c-2 mutants (§3.4): the sub probe's reference reads the winner's derived CTS zone as of the read.

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py engine_v2/plans/plan_w_inputs/mutants_9c2.py

Killers: `tests/test_first_trigger_migration.py::...test_primitive_called_with_the_winners_as_of_cts_zones_and_shared_
frame`, `...test_an_updated_winner_reads_no_derived_zone`; `tests/test_reversal_resolver.py::test_cts_kl_zones_as_of_is_
the_projection_as_of_that_moment`, `::test_the_reversal_reference_reads_the_derived_zone_as_of_the_reversal`. LF anchors.
"""
EDM = "engine_v2/multitf/entity_df_mutation.py"
_SIB = ("    kl_zones = (_cts_kl_zones_as_of(sub_of_event[id(winner)], hi)\n"
        "                if winner is not None and winner.type == \"CTS_CONFIRMED\" else [])\n")
_REV = ("    kl_zones = (_cts_kl_zones_as_of(sub, int(reversal_idx))\n"
        "                if winner is not None and winner.type == \"CTS_CONFIRMED\" else [])\n")
_CLIP = "                bounded.df, clip_events_to_window(bounded.events, cap_local),\n"

MUT = [
    ("v9c2_sibling_ad_hoc", EDM, _SIB, "    kl_zones = []\n"),                                   # the old reader
    ("v9c2_sibling_as_of_lo", EDM, _SIB, _SIB.replace("], hi)", "], lo)")),                       # the wrong moment
    ("v9c2_sibling_any_winner", EDM, _SIB,
     "    kl_zones = _cts_kl_zones_as_of(sub_of_event[id(winner)], hi) if winner is not None else []\n"),
    ("v9c2_reversal_ad_hoc", EDM, _REV, "    kl_zones = []\n"),
    ("v9c2_no_clip", EDM, _CLIP, "                bounded.df, bounded.events,\n"),                # look-ahead
    ("v9c2_no_cap", EDM, "struct_direction=int(sub.direction), lifecycle_cap=cap_local, cap_reason=\"probe_reference\",",
     "struct_direction=int(sub.direction),"),
    ("v9c2_bos_too", EDM, "    return [z for z in zones if z.source_kind == \"CTS\"]\n", "    return list(zones)\n"),
]
