"""PLAN_L landing 2 (Step 3, the ranking rules) — mutants for `mutant_runner.py` (full suite each).

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \
        engine_v2/plans/plan_h_inputs/mutants_l2.py --jobs 14 --out <a short %TEMP% dir>

Every one must be KILLED. Equivalent mutants left out (each reasoned): ranking rule 3's `include_pinbar=False` -> True
(rule 1 already removed every pinbar, so the flag cannot change a result — the False is a guard: a pinbar there
raises, which kills N01); rule 3's runner-up `max(prefer_candle_type(losers, ...))` -> `max(losers)` (the single maru
wins, so every loser is a normal — the same list); `"ip_rank": None if ranked else pick.ip_rank` -> `pick.ip_rank`
and the same for the three `pattern_*_idx` read without `int()` (a no-pattern `IPResult` holds None there — the
explicit None is a guard; the counts, which it does NOT hold as None, are N27). N55 (the landing review's "all pinbars ⇒
no_pattern" assert) was dropped with its assert: it is implied by N56's and the no-IC-candidate ⇔ None assert.
"""
PZ = "engine_v2/zones/poi_zones.py"
EZ = "engine_v2/debug/export_zones.py"
EDM = "engine_v2/multitf/entity_df_mutation.py"
ZR = "engine_v2/charting/_zone_render.py"

MUT = [
    # ---- the ranking rules (`apply_ranking_rules`) ----
    ("N01_rule1_dropped", PZ,
     "    in_play = [i for i in by_idx if str(df.loc[i, \"candle_type\"]) != \"pinbar\"]\n    if not in_play:",
     "    in_play = list(by_idx)\n    if not in_play:"),
    ("N02_rule1_inverted", PZ,
     "    in_play = [i for i in by_idx if str(df.loc[i, \"candle_type\"]) != \"pinbar\"]\n    if not in_play:",
     "    in_play = [i for i in by_idx if str(df.loc[i, \"candle_type\"]) == \"pinbar\"]\n    if not in_play:"),
    ("N03_all_pinbar_picks_one", PZ,
     "    if not in_play:\n        return None, None\n    if len(in_play) == 1:",
     "    if not in_play:\n        return by_idx[min(by_idx)].pattern, None\n    if len(in_play) == 1:"),
    ("N04_icr10_always_exclude_pinbar", PZ,
     "        ranking = None if len(by_idx) == 1 else (RANKING_EXCLUDE_PINBAR, None, len(by_idx), None)",
     "        ranking = (RANKING_EXCLUDE_PINBAR, None, len(by_idx), None)"),
    ("N05_icr10_never_exclude_pinbar", PZ,
     "        ranking = None if len(by_idx) == 1 else (RANKING_EXCLUDE_PINBAR, None, len(by_idx), None)",
     "        ranking = None"),
    ("N06_rule1_count_after", PZ,
     "        ranking = None if len(by_idx) == 1 else (RANKING_EXCLUDE_PINBAR, None, len(by_idx), None)",
     "        ranking = None if len(by_idx) == 1 else (RANKING_EXCLUDE_PINBAR, None, len(in_play), None)"),
    ("N07_rule1_runner_up_computed", PZ,
     "        ranking = None if len(by_idx) == 1 else (RANKING_EXCLUDE_PINBAR, None, len(by_idx), None)",
     "        ranking = None if len(by_idx) == 1 else (RANKING_EXCLUDE_PINBAR, min(by_idx), len(by_idx), None)"),
    ("N08_rule2_skipped", PZ,
     "    passing = filter_immediate_unfilled_imbalance(in_play, immediate)\n    if len(passing) == 1:",
     "    passing = []\n    if len(passing) == 1:"),
    ("N09_rule2_runner_up_oldest", PZ,
     "        runner_up = max(i for i in in_play if i != pick)\n"
     "        return by_idx[pick].pattern, (RANKING_IMMEDIATE_UNFILLED_IMBALANCE",
     "        runner_up = min(i for i in in_play if i != pick)\n"
     "        return by_idx[pick].pattern, (RANKING_IMMEDIATE_UNFILLED_IMBALANCE"),
    ("N10_rule2_count_passing", PZ,
     "(RANKING_IMMEDIATE_UNFILLED_IMBALANCE, int(runner_up), len(in_play),",
     "(RANKING_IMMEDIATE_UNFILLED_IMBALANCE, int(runner_up), len(passing),"),
    ("N11_rule2_gap_dropped", PZ,
     "                                      int(immediate[pick]))", "                                      None)"),
    ("N12_rule2_two_or_more_not_filtered", PZ,
     "    in_play = passing or in_play\n\n    # 3. The candle type, maru > normal.",
     "    in_play = in_play\n\n    # 3. The candle type, maru > normal."),
    ("N13_rule3_skipped", PZ,
     "    by_type = prefer_candle_type(in_play, df, include_pinbar=False)\n    if len(by_type) == 1:",
     "    by_type = prefer_candle_type(in_play, df, include_pinbar=False)\n    if False:"),
    ("N14_rule3_runner_up_oldest_normal", PZ,
     "        runner_up = max(prefer_candle_type(losers, df, include_pinbar=False))",
     "        runner_up = min(prefer_candle_type(losers, df, include_pinbar=False))"),
    ("N15_rule3_count_survivors", PZ,
     "(RANKING_IC_CANDLE_TYPE, int(runner_up), len(in_play), None)",
     "(RANKING_IC_CANDLE_TYPE, int(runner_up), len(by_type), None)"),
    ("N16_rule4_over_rule3_input", PZ,
     "    pick, reference_c2 = most_recent_before_imbalance(by_type, df, fib_state, config)\n"
     "    runner_up = max(i for i in by_type if i != pick)\n    return by_idx[pick].pattern, (RANKING_",
     "    pick, reference_c2 = most_recent_before_imbalance(in_play, df, fib_state, config)\n"
     "    runner_up = max(i for i in by_type if i != pick)\n    return by_idx[pick].pattern, (RANKING_"),
    ("N17_rule4_runner_up_over_rule3_input", PZ,
     "    runner_up = max(i for i in by_type if i != pick)\n    return by_idx[pick].pattern, (RANKING_",
     "    runner_up = max(i for i in in_play if i != pick)\n    return by_idx[pick].pattern, (RANKING_"),
    ("N18_rule4_count_in_play", PZ,
     "(RANKING_MOST_RECENT_BEFORE_IMBALANCE, int(runner_up), len(by_type),",
     "(RANKING_MOST_RECENT_BEFORE_IMBALANCE, int(runner_up), len(in_play),"),
    ("N19_rule4_gap_dropped", PZ,
     "                                  int(reference_c2))", "                                  None)"),
    # ---- `select_ic`: the wiring ----
    ("N20_step3_not_called", PZ,
     "    pick, ranking = apply_ranking_rules(df, fib_state, config, candidate_checks)",
     "    pick, ranking = None, None"),
    ("N21_step3_over_every_check", PZ,
     "    pick, ranking = apply_ranking_rules(df, fib_state, config, candidate_checks)",
     "    pick, ranking = apply_ranking_rules(df, fib_state, config, checks)"),
    ("N22_step3_also_with_a_pattern", PZ,
     "    if pick is not None:\n        return ICSelection(checks, pick, tiebreak, ip_outcome=IP_OUTCOME_PATTERN",
     "    if False:\n        return ICSelection(checks, pick, tiebreak, ip_outcome=IP_OUTCOME_PATTERN"),
    ("N23_pattern_outcome_mislabelled", PZ,
     "        return ICSelection(checks, pick, tiebreak, ip_outcome=IP_OUTCOME_PATTERN, ranking=None, no_ic_reason=None)",
     "        return ICSelection(checks, pick, tiebreak, ip_outcome=IP_OUTCOME_NO_PATTERN, ranking=None,"
     " no_ic_reason=None)"),
    ("N24_all_pinbar_reason_no_candidate", PZ,
     "                       no_ic_reason=None if pick is not None else ALL_IC_CANDIDATES_PINBAR)",
     "                       no_ic_reason=None if pick is not None else NO_IC_CANDIDATE)"),
    ("N25_post_init_pick_reason_assert_dropped", PZ,
     "        assert (self.pick is None) == (self.no_ic_reason is not None), (",
     "        assert True, ("),
    # ---- `derive_poi_zones`: the meta, the floor, the assert ----
    ("N26_ranked_flag_never", PZ,
     "        ranked = rec.selection.ip_outcome == IP_OUTCOME_NO_PATTERN",
     "        ranked = False"),
    ("N27_counts_not_nulled", PZ,
     "                \"ip_leg_candle_counts\": None if ranked else pick.ip_leg_candle_counts,",
     "                \"ip_leg_candle_counts\": pick.ip_leg_candle_counts,"),
    ("N28_moment_assert_dropped", PZ,
     "        assert (pick.pattern_confirmed_idx is None) == ranked, (",
     "        assert True, ("),
    ("N29_floor_pattern_term_dropped", PZ,
     "    if pattern_confirmed_idx is not None:\n        first_active = max(first_active, int(pattern_confirmed_idx))\n",
     ""),
    ("N30_meta_ranking_dropped", PZ,
     "        rk = rec.selection.ranking or (None, None, None, None)\n        zone = POIZone(",
     "        rk = (None, None, None, None)\n        zone = POIZone("),
    ("N31_meta_count_is_runner_up", PZ,
     "                \"ranking_in_play_count\": rk[2],", "                \"ranking_in_play_count\": rk[1],"),
    ("N32_meta_outcome_constant", PZ,
     "                \"ip_outcome\": rec.selection.ip_outcome,", "                \"ip_outcome\": \"pattern\","),
    # ---- `derive_ic_candidates`: the CSV rows ----
    ("N33_ranking_on_every_row", PZ,
     "            rk = sel.ranking if picked and sel.ranking is not None else (None, None, None, None)",
     "            rk = sel.ranking if sel.ranking is not None else (None, None, None, None)"),
    ("N34_outcome_on_the_pick_only", PZ,
     "                is_ic_candidate=bool(c.is_ic_candidate),\n                ip_outcome=sel.ip_outcome,",
     "                is_ic_candidate=bool(c.is_ic_candidate),\n                ip_outcome=sel.ip_outcome if picked else None,"),
    ("N35_no_ic_reason_constant", PZ,
     "                reason=sel.no_ic_reason,", "                reason=\"no_ic_candidate\","),
    ("N36_no_ic_row_outcome_dropped", PZ,
     "                pattern_confirmed_idx=None, ip_leg_candle_counts=None, picked=False,\n"
     "                ip_outcome=sel.ip_outcome,\n",
     "                pattern_confirmed_idx=None, ip_leg_candle_counts=None, picked=False,\n"),
    ("N37_csv_ranking_columns_missing", EZ,
     "    \"ranking_rule\", \"ranking_runner_up_ic_idx\", \"ranking_in_play_count\", \"ranking_imbalance_first_c2_idx\",\n]",
     "]"),
    ("N38_csv_gap_cell_is_runner_up", EZ,
     "            \"ranking_imbalance_first_c2_idx\": r.ranking_imbalance_first_c2_idx,",
     "            \"ranking_imbalance_first_c2_idx\": r.ranking_runner_up_ic_idx,"),
    ("N39_csv_outcome_cell_empty", EZ,
     "            \"ip_outcome\": r.ip_outcome,", "            \"ip_outcome\": None,"),
    # ---- the mirror ----
    ("N40_zone_keys_drop_runner_up", EDM,
     "    \"ranking_runner_up_ic_idx\",       # POI (None unless a ranking rule 2–4 decided)\n", ""),
    ("N41_ic_fields_drop_gap", EDM,
     "    \"ranking_runner_up_ic_idx\",         # PLAN_L landing 2\n    \"ranking_imbalance_first_c2_idx\",\n)",
     "    \"ranking_runner_up_ic_idx\",         # PLAN_L landing 2\n)"),
    # ---- the hovers ----
    ("N42_poi_hover_ranked_branch_dropped", ZR,
     "    if meta[\"ip_outcome\"] == IP_OUTCOME_NO_PATTERN:\n        return _poi_ranking_hover_text(meta, ic_idx)\n", ""),
    ("N43_poi_hover_only_text", ZR,
     "        return \"No pattern · only IC candidate\"", "        return \"No pattern\""),
    ("N44_poi_hover_runner_up_dropped", ZR,
     "        text += f\" over {int(runner_up)}\"", "        pass"),
    ("N45_poi_hover_parts_type_has_gap", ZR,
     "    RANKING_IC_CANDLE_TYPE: (True, False),", "    RANKING_IC_CANDLE_TYPE: (True, True),"),
    ("N46_poi_hover_pattern_assert_dropped", ZR,
     "    assert pattern == [None] * len(pattern), f\"[chart] ranked POI {ic} carries pattern meta {pattern}\"",
     "    assert True"),
    ("N47_poi_hover_names_the_ic", ZR,
     "    text = f\"No pattern · ranking rule {rule}\"", "    text = f\"No pattern · IC {ic} · ranking rule {rule}\""),
    ("N48_fib_hover_ranked_branch_dropped", ZR,
     "    elif ic_row.ip_outcome == IP_OUTCOME_NO_PATTERN:\n", "    elif False:\n"),
    ("N49_fib_hover_ip_none_label", ZR,
     "        ip = f\"IC: none ({ic_row.reason})\"", "        ip = f\"IP: none ({ic_row.reason})\""),
    ("N50_fib_hover_drops_the_ic", ZR,
     "        ip = f\"No pattern · IC {int(ic_row.ic_idx)} · {how}\"", "        ip = f\"No pattern · {how}\""),
    ("N51_fib_hover_outcome_assert_dropped", ZR,
     "        assert ic_row.ip_outcome == IP_OUTCOME_PATTERN, (\n", "        assert True, (\n"),
    # ---- the landing review (2026-10-10, lens A): the runner-ups' lists, the selection's asserts ----
    ("N53_rule2_runner_up_from_every_candidate", PZ,
     "        runner_up = max(i for i in in_play if i != pick)\n"
     "        return by_idx[pick].pattern, (RANKING_IMMEDIATE_UNFILLED_IMBALANCE",
     "        runner_up = max(i for i in by_idx if i != pick)\n"
     "        return by_idx[pick].pattern, (RANKING_IMMEDIATE_UNFILLED_IMBALANCE"),
    ("N54_rule3_losers_from_every_non_pinbar", PZ,
     "        losers = [i for i in in_play if i != by_type[0]]\n"
     "        runner_up = max(prefer_candle_type(losers, df, include_pinbar=False))",
     "        losers = [i for i in by_idx if i != by_type[0] and str(df.loc[i, \"candle_type\"]) != \"pinbar\"]\n"
     "        runner_up = max(prefer_candle_type(losers, df, include_pinbar=False))"),
    ("N56_post_init_pattern_rank_dropped", PZ,
     "        assert self.ip_outcome != IP_OUTCOME_PATTERN or (self.pick is not None and self.pick.rank is not None), (",
     "        assert True, ("),
    ("N57_post_init_ranked_rank_dropped", PZ,
     "        assert self.ip_outcome != IP_OUTCOME_NO_PATTERN or self.pick is None or self.pick.rank is None, (",
     "        assert True, ("),
    # ---- MS's read ----
    ("N52_ms_ignores_a_ranked_pick", PZ,
     "    if sel.pick is None:\n        return None\n    ic_idx = int(sel.pick.ic_idx)",
     "    if sel.pick is None or sel.ip_outcome != IP_OUTCOME_PATTERN:\n        return None\n"
     "    ic_idx = int(sel.pick.ic_idx)"),
]
