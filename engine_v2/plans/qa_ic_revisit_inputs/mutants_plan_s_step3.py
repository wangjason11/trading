"""PLAN_S step 3 mutants (§4.5) — Step 1B, the prior-imbalance preference: each must fail the suite.

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py engine_v2/plans/qa_ic_revisit_inputs/mutants_plan_s_step3.py

NOT mutants (equivalent — plan review lens B): "closest" by the largest `start_idx` (instances never overlap) and
`end_idx <= ic_idx` (the IC can never be an sd c2). Dropping the `len(passers) < len(candidate_checks)` guard of the
outranked computation is equivalent too (with every candidate a passer, Step 2's pick over all is a passer).
"""
P = "engine_v2/zones/poi_zones.py"
R = "engine_v2/charting/_zone_render.py"
E = "engine_v2/multitf/entity_df_mutation.py"
X = "engine_v2/debug/export_zones.py"

_GATE = "        if inst.direction != sd or inst.start_idx < fib_state.bos_idx or inst.end_idx >= ic_idx:\n"
_FILL = "        prior_imbalance_unfilled=not inst.is_filled(df, check.ic_idx - 1, config.fill_threshold),\n"
_PASSERS = "    steps = _steps_2_and_3(df, fib_state, config, passers) if passers else None\n"
_FALLBACK = "        steps = _steps_2_and_3(df, fib_state, config, candidate_checks)\n"

MUT = [
    # --- the gap (`closest_prior_imbalance`) ---
    ("gap_bos_strict", P, _GATE, _GATE.replace("inst.start_idx < fib_state.bos_idx", "inst.start_idx <= fib_state.bos_idx")),
    ("gap_bos_dropped", P, _GATE, _GATE.replace(" or inst.start_idx < fib_state.bos_idx", "")),
    ("gap_c3_is_ic_excluded", P, _GATE, _GATE.replace("inst.end_idx >= ic_idx", "inst.end_idx >= ic_idx - 1")),
    ("gap_direction_dropped", P, _GATE, _GATE.replace("inst.direction != sd or ", "")),
    ("gap_earliest", P, "        if best is None or inst.end_idx > best.end_idx:\n",
     "        if best is None or inst.end_idx < best.end_idx:\n"),
    # --- the test (`check_prior_imbalance`) ---
    ("fill_horizon_ic", P, _FILL, _FILL.replace("check.ic_idx - 1", "check.ic_idx")),
    ("fill_horizon_ic_minus_2", P, _FILL, _FILL.replace("check.ic_idx - 1", "check.ic_idx - 2")),
    ("fill_threshold_dropped", P, _FILL, _FILL.replace(", config.fill_threshold)", ")")),
    ("no_gap_passes", P,
     "        return replace(check, prior_imbalance_first_c2_idx=None, prior_imbalance_unfilled=False)\n",
     "        return replace(check, prior_imbalance_first_c2_idx=None, prior_imbalance_unfilled=True)\n"),
    ("gap_named_by_its_last_c2", P, "        prior_imbalance_first_c2_idx=int(inst.start_idx),\n",
     "        prior_imbalance_first_c2_idx=int(inst.end_idx),\n"),
    # --- the flow (`select_ic`) ---
    ("passers_ignored", P, _PASSERS, _PASSERS.replace("config, passers)", "config, candidate_checks)")),
    ("passers_fed_every_check", P, _PASSERS, _PASSERS.replace("config, passers)", "config, checks)")),
    ("fallback_fed_every_check", P, _FALLBACK, _FALLBACK.replace("candidate_checks)", "checks)")),
    ("fallback_dropped", P, "    if steps is not None and steps[0] is not None:\n", "    if steps is not None:\n"),
    ("fallback_outcomes_swapped", P,
     "        outcome = PRIOR_IMBALANCE_FALLBACK_PASSERS_NO_IC if passers else PRIOR_IMBALANCE_FALLBACK_NO_PASSER\n",
     "        outcome = PRIOR_IMBALANCE_FALLBACK_NO_PASSER if passers else PRIOR_IMBALANCE_FALLBACK_PASSERS_NO_IC\n"),
    ("outcome_value_renamed", P, 'PRIOR_IMBALANCE_PREFERRED = "preferred"', 'PRIOR_IMBALANCE_PREFERRED = "preferred_"'),
    ("only_candidates_with_classify_all", P,
     "        check_prior_imbalance(df, c, fib_state, config) if (classify_all or c.ic_idx in ic_candidates) else c\n",
     "        check_prior_imbalance(df, c, fib_state, config) if c.ic_idx in ic_candidates else c\n"),
    ("outranked_never", P, "                outranked = best\n", "                outranked = None\n"),
    ("outranked_even_a_passer", P,
     "            if best is not None and best.ic_idx not in {c.ic_idx for c in passers}:\n",
     "            if best is not None:\n"),
    # --- the counts (`ICSelection` properties) ---
    ("candidate_count_off_by_one", P, "        return None if self.prior_imbalance_outcome is None else len(self.ic_candidates)\n",
     "        return None if self.prior_imbalance_outcome is None else len(self.ic_candidates) + 1\n"),
    ("passer_count_is_candidates", P,
     "        return sum(1 for c in self.ic_candidates if c.prior_imbalance_unfilled)\n",
     "        return len(self.ic_candidates)\n"),
    # --- the consumers ---
    ("poi_meta_gap_of_the_wrong_check", P,
     '                "prior_imbalance_first_c2_idx": rec.selection.pick_check.prior_imbalance_first_c2_idx,\n',
     '                "prior_imbalance_first_c2_idx": rec.selection.checks[0].prior_imbalance_first_c2_idx,\n'),
    ("poi_meta_outcome_missing", P, '                "prior_imbalance_outcome": rec.selection.prior_imbalance_outcome,\n', ""),
    ("csv_record_values_not_on_the_no_ic_row", P,
     "                ic_zone_band=sel.ic_zone_band,\n                **prior,\n                meta=",
     "                ic_zone_band=sel.ic_zone_band,\n                meta="),
    ("csv_column_missing", X, '            "prior_imbalance_unfilled": r.prior_imbalance_unfilled,\n', ""),
    ("mirror_csv_idx_missing", E, '    "prior_imbalance_first_c2_idx",     # PLAN_S step 3 (Step 1B, every check row)\n', ""),
    ("mirror_meta_outranked_missing", E,
     '    "prior_imbalance_outranked_ic_idx",  # POI (the pattern the preference outranked; None unless it did)\n', ""),
    # --- the hovers ---
    ("narrowed_le", P, "    return outcome == PRIOR_IMBALANCE_PREFERRED and int(passer_count) < int(candidate_count)\n",
     "    return outcome == PRIOR_IMBALANCE_PREFERRED and int(passer_count) <= int(candidate_count)\n"),
    ("narrowed_outcome_dropped", P,
     "    return outcome == PRIOR_IMBALANCE_PREFERRED and int(passer_count) < int(candidate_count)\n",
     "    return outcome is not None and int(passer_count) < int(candidate_count)\n"),
    ("only_candidate_wording", R, '    return "only preferred IC candidate" if narrowed else "only IC candidate"\n',
     '    return "only IC candidate"\n'),
    ("fib_hover_not_narrowed", R,
     "        how = _only_candidate_text(narrowed) if ic_row.ranking_rule is None else f\"ranking rule {ic_row.ranking_rule}\"\n",
     "        how = _only_candidate_text(False) if ic_row.ranking_rule is None else f\"ranking rule {ic_row.ranking_rule}\"\n"),
    ("hover_suffix_dropped", R, '        text += f" (over pattern {outranked_rank} at {int(outranked_ic)})"\n',
     "        pass\n"),
    ("hover_counts_swapped", R,
     "    text = (f\"<br>Prior imbalance {int(c2)} unfilled · preferred {int(meta['prior_imbalance_passer_count'])} of \"\n"
     "            f\"{int(meta['ic_candidate_count'])} IC candidates\")\n",
     "    text = (f\"<br>Prior imbalance {int(c2)} unfilled · preferred {int(meta['ic_candidate_count'])} of \"\n"
     "            f\"{int(meta['prior_imbalance_passer_count'])} IC candidates\")\n"),
    ("hover_line_after_the_band", R,
     "    tail = _poi_prior_imbalance_hover_text(meta, ic_idx)            # Step 1B's line, then the band's\n"
     '    tail += "<br>IC zone 80-100 (fallback)" if meta["ic_zone_band"] == IC_ZONE_FALLBACK else ""\n',
     '    tail = "<br>IC zone 80-100 (fallback)" if meta["ic_zone_band"] == IC_ZONE_FALLBACK else ""\n'
     "    tail += _poi_prior_imbalance_hover_text(meta, ic_idx)\n"),
    # --- the invariant lens B asked for ---
    ("pick_not_a_candidate_unguarded", P,
     "        assert self.pick is None or int(self.pick.ic_idx) in candidates, (\n",
     "        assert True or int(self.pick.ic_idx) in candidates, (\n"),
]

# The landing review lens 2's mutants (2026-10-21): N1–N6 SURVIVED the first landing (guard asserts without a lone pin,
# and no FILLED prior gap end to end) — each fixed by a test; N7 was killed already.
MUT += [
    ("N1_noic_vs_ipoutcome_assert_dropped", P,
     "        assert (self.no_ic_reason == NO_IC_CANDIDATE) == (self.ip_outcome is None), (\n",
     "        assert True or (self.no_ic_reason == NO_IC_CANDIDATE) == (self.ip_outcome is None), (\n"),
    ("N2_ranking_requires_pick_dropped", P,
     "        assert self.ranking is None or (self.ip_outcome == IP_OUTCOME_NO_PATTERN and self.pick is not None), (\n",
     "        assert self.ranking is None or (self.ip_outcome == IP_OUTCOME_NO_PATTERN), (\n"),
    ("N3_every_candidate_tested_dropped", P,
     "        assert all(c.prior_imbalance_unfilled is not None for c in candidates.values()) or self.ip_outcome is None, (\n",
     "        assert True or all(c.prior_imbalance_unfilled is not None for c in candidates.values()) or self.ip_outcome is None, (\n"),
    ("N4_passer_gap_before_ic_dropped", P,
     "                assert c.prior_imbalance_first_c2_idx is not None and c.prior_imbalance_first_c2_idx < c.ic_idx, c\n",
     "                assert c.prior_imbalance_first_c2_idx is not None, c\n"),
    ("N5_fallback_passers_nonempty_dropped", P,
     "            assert passers and (self.pick is None or int(self.pick.ic_idx) not in passers), (\n",
     "            assert (self.pick is None or int(self.pick.ic_idx) not in passers), (\n"),
    ("N6_csv_unfilled_is_has_gap", P,
     "                prior_imbalance_unfilled=bool(c.prior_imbalance_unfilled),\n",
     "                prior_imbalance_unfilled=c.prior_imbalance_first_c2_idx is not None,\n"),
    ("N7_ms_tests_every_check", P,
     "        check_prior_imbalance(df, c, fib_state, config) if (classify_all or c.ic_idx in ic_candidates) else c\n",
     "        check_prior_imbalance(df, c, fib_state, config)\n"),
]

