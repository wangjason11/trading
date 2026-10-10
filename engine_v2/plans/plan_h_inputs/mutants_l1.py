"""PLAN_L landing 1 (the IC identification in steps — the restructure) — mutants for `mutant_runner.py` (full suite
each).

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \
        engine_v2/plans/plan_h_inputs/mutants_l1.py --jobs 14 --out <a short %TEMP% dir>

Every one must be KILLED.
"""
PZ = "engine_v2/zones/poi_zones.py"
EZ = "engine_v2/debug/export_zones.py"

MUT = [
    # Step 1 — the predicate: every condition counts
    ("L01_pred_drops_cond3", PZ,
     "        return bool(self.has_unfilled_imbalance_after) and self.ic_ref_reason is None and bool(self.touches_zone)",
     "        return self.ic_ref_reason is None and bool(self.touches_zone)"),
    ("L02_pred_drops_touch", PZ,
     "        return bool(self.has_unfilled_imbalance_after) and self.ic_ref_reason is None and bool(self.touches_zone)",
     "        return bool(self.has_unfilled_imbalance_after) and self.ic_ref_reason is None"),
    ("L03_pred_drops_reference", PZ,
     "        return bool(self.has_unfilled_imbalance_after) and self.ic_ref_reason is None and bool(self.touches_zone)",
     "        return bool(self.has_unfilled_imbalance_after) and bool(self.touches_zone)"),
    # Step 1 — the checks: the row set, the window, each outcome
    ("L04_rowset_candle_direction", PZ,
     "        if int(agreed_s.loc[idx]) != -sd:",
     "        if int(df.loc[idx, \"direction\"]) != -sd:"),
    ("L05_window_excludes_cts", PZ,
     "    agreed_s = pd.Series(agreed, index=df.index)\n    checks = []\n"
     "    for idx in range(fib_state.bos_idx, fib_state.cts_idx + 1):",
     "    agreed_s = pd.Series(agreed, index=df.index)\n    checks = []\n"
     "    for idx in range(fib_state.bos_idx, fib_state.cts_idx):"),
    ("L06_cond3_starts_late", PZ,
     "        start_idx=idx + 1,        # strictly after", "        start_idx=idx + 2,        # strictly after"),
    ("L07_touch_always_true", PZ,
     "            touches_zone=ic_touches_zone(float(df.loc[idx, \"h\"]), float(df.loc[idx, \"l\"]), zone_lo, zone_hi),",
     "            touches_zone=True,"),
    ("L08_reference_ignored", PZ,
     "            ic_ref_reason=ic_reference_reason(df, idx, ic_reference, sd),",
     "            ic_ref_reason=None,"),
    ("L09_cond3_always_true", PZ,
     "            has_unfilled_imbalance_after=bool(_has_unfilled_imbalance_after(df, idx, fib_state, config)),",
     "            has_unfilled_imbalance_after=True,"),
    # the classification: classify_all vs the IC candidates only; the pick reads the IC candidates only
    ("L10_classify_all_ignored", PZ,
     "    to_classify = [c.ic_idx for c in checks if classify_all or c.is_ic_candidate]",
     "    to_classify = [c.ic_idx for c in checks if c.is_ic_candidate]"),
    ("L11_ms_classifies_everything", PZ,
     "    to_classify = [c.ic_idx for c in checks if classify_all or c.is_ic_candidate]",
     "    to_classify = [c.ic_idx for c in checks]"),
    ("L12_step2_reads_every_classified_check", PZ,
     "        df, fib_state, config, tuple(c for c in checks if c.is_ic_candidate))",
     "        df, fib_state, config, tuple(c for c in checks if c.pattern is not None))"),
    ("L13_immediate_fact_not_set", PZ,
     "                immediate_imbalance_first_c2_idx=immediate_unfilled_imbalance_first_c2_idx(",
     "                immediate_imbalance_first_c2_idx=None and immediate_unfilled_imbalance_first_c2_idx("),
    ("L14_find_returns_every_check", PZ,
     "    return [c.ic_idx for c in checks if c.is_ic_candidate]",
     "    return [c.ic_idx for c in checks]"),
    # the shared rule helpers
    ("L15_immediate_filter_inverted", PZ,
     "    return [i for i in in_play if immediate_c2[i] is not None]",
     "    return [i for i in in_play if immediate_c2[i] is None]"),
    ("L16_type_keeps_pinbar_when_excluded", PZ,
     "    order = IC_TYPE_PREFERENCE if include_pinbar else tuple(t for t in IC_TYPE_PREFERENCE if t != \"pinbar\")",
     "    order = IC_TYPE_PREFERENCE"),
    ("L17_type_prefers_worst", PZ,
     "    best = min(type_rank.values())", "    best = max(type_rank.values())"),
    ("L18_most_recent_picks_oldest", PZ,
     "    pick = max(i for i in in_play if i < reference.start_idx)",
     "    pick = min(i for i in in_play if i < reference.start_idx)"),
    ("L19_reference_after_newest", PZ,
     "    oldest = min(in_play)\n    reference = min(",
     "    oldest = max(in_play)\n    reference = min("),
    ("L20_type_runner_up_any_loser", PZ,
     "        runner_up = max(prefer_candle_type(losers, df, include_pinbar=True))",
     "        runner_up = max(losers)"),
    ("L21_window_param_3", PZ,
     "    immediate_imbalance_candles: int = 2", "    immediate_imbalance_candles: int = 3"),
    # the debug records + the CSV
    ("L22_record_cond3_true", PZ,
     "                has_unfilled_imbalance_after=bool(c.has_unfilled_imbalance_after),",
     "                has_unfilled_imbalance_after=True,"),
    ("L23_record_candidate_true", PZ,
     "                is_ic_candidate=bool(c.is_ic_candidate),", "                is_ic_candidate=True,"),
    ("L24_none_reason_from_checks", PZ,
     "                reason=(\"no_pattern\" if sel.ic_candidates else \"no_ic_candidate\"),",
     "                reason=(\"no_pattern\" if sel.checks else \"no_ic_candidate\"),"),
    ("L25_csv_column_missing", EZ,
     "    \"has_unfilled_imbalance_after\", \"is_ic_candidate\",\n",
     "    \"is_ic_candidate\",\n"),
    ("L26_csv_cell_constant", EZ,
     "            \"has_unfilled_imbalance_after\": r.has_unfilled_imbalance_after,",
     "            \"has_unfilled_imbalance_after\": True,"),
    # the fail-loud frame read (before the guard) and the V path's finder
    ("L27_unstamped_frame_defaults", PZ,
     "    timeframe = df.attrs[\"timeframe\"]\n    params = candle_params_for_timeframe(timeframe)\n"
     "    cap = config.ip_leg_cap_by_tf[timeframe]\n    fib = fib_state.fib\n",
     "    timeframe = df.attrs.get(\"timeframe\", \"H1\")\n    params = candle_params_for_timeframe(timeframe)\n"
     "    cap = config.ip_leg_cap_by_tf[timeframe]\n    fib = fib_state.fib\n"),
    ("L28_v_finder_drops_cond3", PZ,
     "        if not _has_unfilled_imbalance_after(df, idx, fib_state, config):\n            continue\n"
     "        candidates.append(idx)",
     "        candidates.append(idx)"),
    # the landing review (2026-10-09): the guard's one holder, tiebreak rule 3's runner-up
    ("R08_guard_drops_locked_inactive", PZ,
     "    if fib is None or not (fib_state.active or fib_state.locked):\n        return ICSelection((), None, None)",
     "    if fib is None or not fib_state.active:\n        return ICSelection((), None, None)"),
    ("R10_guard_dropped", PZ,
     "    if fib is None or not (fib_state.active or fib_state.locked):\n        return ICSelection((), None, None)",
     "    if fib is None:\n        return ICSelection((), None, None)"),
    ("R02_rule3_runner_up_over_rule2_input", PZ,
     "    runner_up = max(i for i in by_type if i != pick)",
     "    runner_up = max(i for i in in_play if i != pick)"),
]
