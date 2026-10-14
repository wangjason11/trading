"""PLAN_P (IC condition 3 = the immediate unfilled imbalance + Step 1A, the 80–100 fallback zone) — mutants for
`mutant_runner.py` (full suite each; PLAN_P §7 + the plan review's additions).

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \
        engine_v2/plans/qa_ic_crossfib_inputs/mutants_plan_p.py --out <a short %TEMP% dir>

Every one must be KILLED except E01 (EQUIVALENT: an immediate gap implies `has_unfilled_imbalance_after` — the same sd
filter and fill horizon over a sub-window).
"""
PZ = "engine_v2/zones/poi_zones.py"
ZR = "engine_v2/charting/_zone_render.py"
EZ = "engine_v2/debug/export_zones.py"

_CAND = "return (self.immediate_imbalance_first_c2_idx is not None and self.ic_ref_reason is None"
_STEP1A_IF = "if checks and not find_ic_candidates(checks):"
_STEP1A_BAND = ("band = IC_ZONE_FALLBACK\n"
                "        zone_lo, zone_hi = fib_zone_bounds(fib.anchor_high, fib.anchor_low, sd, ic_zone_config(config, band))")
_STEP1A_TOUCH = "replace(c, touches_zone=ic_touches_zone("
_POI_BAND = 'band = "<br>IC zone 80-100 (fallback)" if meta["ic_zone_band"] == IC_ZONE_FALLBACK else ""'

MUT = [
    # condition 3
    ("P01_cond3_reads_the_retired_any_gap", PZ, _CAND,
     "return (bool(self.has_unfilled_imbalance_after) and self.ic_ref_reason is None"),
    ("P02_cond3_dropped", PZ, _CAND, "return (True and self.ic_ref_reason is None"),
    ("P03_cond3_window_unclipped_at_the_cts", PZ,
     "hi = min(ic_idx + config.immediate_imbalance_candles, int(fib_state.cts_idx))",
     "hi = ic_idx + config.immediate_imbalance_candles"),
    # Step 1A
    ("P04_step_1a_never_runs", PZ, _STEP1A_IF, "if False:"),
    ("P05_step_1a_also_after_an_all_pinbar_list", PZ, _STEP1A_IF,
     'if checks and not [i for i in find_ic_candidates(checks) if str(df.loc[i, "candle_type"]) != "pinbar"]:'),
    ("P06_step_1a_on_the_primary_zone", PZ,
     "zone_lo, zone_hi = fib_zone_bounds(fib.anchor_high, fib.anchor_low, sd, ic_zone_config(config, band))",
     "zone_lo, zone_hi = fib_zone_bounds(fib.anchor_high, fib.anchor_low, sd, config)"),
    ("P07_step_1a_drops_condition_2", PZ, _STEP1A_TOUCH, "replace(c, ic_ref_reason=None, touches_zone=ic_touches_zone("),
    ("P08_step_1a_drops_condition_3", PZ, _STEP1A_TOUCH,
     "replace(c, immediate_imbalance_first_c2_idx=c.ic_idx + 1, touches_zone=ic_touches_zone("),
    ("P09_band_recorded_as_primary", PZ, _STEP1A_BAND,
     "band = IC_ZONE_PRIMARY\n"
     "        zone_lo, zone_hi = fib_zone_bounds(fib.anchor_high, fib.anchor_low, sd, ic_zone_config(config, IC_ZONE_FALLBACK))"),
    ("P10_fallback_min_61_8", PZ, "ic_fallback_fib_min: float = 80.0", "ic_fallback_fib_min: float = 61.8"),
    ("P11_fallback_max_80", PZ, "ic_fallback_fib_max: float = 100.0", "ic_fallback_fib_max: float = 80.0"),
    ("P12_zone_config_keeps_the_primary_min", PZ,
     "return replace(config, ic_fib_min=config.ic_fallback_fib_min, ic_fib_max=config.ic_fallback_fib_max)",
     "return replace(config, ic_fib_min=config.ic_fib_min, ic_fib_max=config.ic_fallback_fib_max)"),
    ("P13_selection_allows_the_fallback_without_checks", PZ,
     'assert self.ic_zone_band == IC_ZONE_PRIMARY or self.checks, "[poi_zones] Step 1A ran without checks"',
     "pass"),
    # the consumers
    ("P14_activation_touch_on_the_primary_band", PZ, "config=ic_zone_config(config, rec.selection.ic_zone_band),",
     "config=config,"),
    ("P15_poi_meta_band_always_primary", PZ, '"ic_zone_band": rec.selection.ic_zone_band,',
     '"ic_zone_band": IC_ZONE_PRIMARY,'),
    ("P16_csv_rows_band_always_primary", PZ,
     "ranking_imbalance_first_c2_idx=rk[3],\n                ic_zone_band=sel.ic_zone_band,",
     "ranking_imbalance_first_c2_idx=rk[3],\n                ic_zone_band=IC_ZONE_PRIMARY,"),
    ("P17_csv_no_ic_row_band_always_primary", PZ,
     "ip_outcome=sel.ip_outcome,\n                ic_zone_band=sel.ic_zone_band,",
     "ip_outcome=sel.ip_outcome,\n                ic_zone_band=IC_ZONE_PRIMARY,"),
    ("P18_csv_column_not_exported", EZ, '"ic_zone_band": r.ic_zone_band,', '"ic_zone_band": None,'),
    ("P19_activation_cond3_only_the_immediate_gap", PZ,
     "and inst.end_idx > ic_idx          # overlaps (ic_idx, t] only when end_idx > ic_idx",
     "and ic_idx < inst.start_idx <= ic_idx + 2"),
    ("P20_most_recent_reference_after_the_newest", PZ, "    oldest = min(in_play)\n", "    oldest = max(in_play)\n"),
    # the hovers
    ("P21_poi_hover_line_missing", ZR, _POI_BAND, 'band = ""'),
    ("P22_poi_hover_line_on_the_primary_band", ZR, _POI_BAND,
     'band = "<br>IC zone 80-100 (fallback)" if meta["ic_zone_band"] != IC_ZONE_FALLBACK else ""'),
    ("P23_poi_hover_pattern_branch_drops_the_line", ZR, "    return text + band\n", "    return text\n"),
    ("P24_poi_hover_ranked_branch_drops_the_line", ZR, "return _poi_ranking_hover_text(meta, ic_idx) + band",
     "return _poi_ranking_hover_text(meta, ic_idx)"),
    ("P25_poi_hover_band_assert_deleted", ZR,
     'assert meta["ic_zone_band"] in (IC_ZONE_PRIMARY, IC_ZONE_FALLBACK), (',
     'assert True or meta["ic_zone_band"] in (IC_ZONE_PRIMARY, IC_ZONE_FALLBACK), ('),
    ("P26_fib_suffix_missing", ZR, 'ip += " · zone 80-100 (fallback)"', "pass"),
    ("P27_fib_band_assert_deleted", ZR, "assert ic_row.ic_zone_band in (IC_ZONE_PRIMARY, IC_ZONE_FALLBACK), (",
     "assert True or ic_row.ic_zone_band in (IC_ZONE_PRIMARY, IC_ZONE_FALLBACK), ("),
    # expected EQUIVALENT
    ("E01_cond3_also_ands_the_retired_any_gap", PZ, _CAND,
     "return (self.immediate_imbalance_first_c2_idx is not None and bool(self.has_unfilled_imbalance_after)"
     " and self.ic_ref_reason is None"),
]
