"""Mutants of PLAN_H landing A (the institutional-pattern selector, exported, nothing switched) for
`plans/plan_e_inputs/review_scripts/mutant_runner.py` — PLAN_H §8.6: every IP rule branch + the call sites.

usage (repo root):
  python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py engine_v2/plans/plan_h_inputs/mutants_plan_h_a.py

Template: `(id, relpath, old, new)`; `old` must occur exactly once in the file. Ids: A agreed direction, L legs,
R ranks / stacks, S selection (touch, TB1, TB2a), G the record gate / rows, M the sub mirror, O / X / C / Q / I / B the
wiring (orchestrator, exporter, classifier stamp, qualified copies, imbalance generator, big flags).
First run 62/66 KILLED; the 4 survivors (L12, S13, O2, B1) got a test each → 66/66. The landing review's test-gap lens
(2026-10-06) added N01–N33 (30 written; 7 killed by the suite then; 22 survivors got a killing test each — the
`test_institutional_pattern.py` block "the landing review's test-gap lens"); N18 (the TB2 reference taken in attrs
order instead of the min `start_idx`) is EQUIVALENT — `compute_imbalance` appends in ascending `start_idx` and every
frame shares that list — and is kept out of the run list.
"""
IP = "engine_v2/patterns/institutional_pattern.py"
POI = "engine_v2/zones/poi_zones.py"
CV2 = "engine_v2/features/candles_v2.py"
CC = "engine_v2/features/candle_classifier.py"
EDM = "engine_v2/multitf/entity_df_mutation.py"
ORCH = "engine_v2/pipeline/orchestrator.py"
EXP = "engine_v2/debug/export_zones.py"
WC = "engine_v2/zones/wave_candles.py"
IMB = "engine_v2/patterns/imbalance.py"

_CONT_RUN = '''    while j <= cts_anchor_idx and agreed[j] in (0, sd):
        run.append(j)'''
_INIT_RUN = "    while j >= bos_anchor_idx and agreed[j] in (0, sd):"
_CONFIRMED = "    confirmed = capped[-1] if len(capped) == cap else capped[-1] + 1"
_INITS = "        inits = [(_trim(seq[-cap:], agreed, lead=True), False)]"
_TB2_PICK = "    pick = max((r for r in tb1 if r.ic_idx < reference.start_idx), key=lambda r: r.ic_idx)"

MUT = [
    # --- A: the agreed direction (V1a) ---
    ("A1_split_is_agreed", CV2, "np.where((v == 0) | (v == d), d, AGREED_SPLIT)", "np.where(True, d, AGREED_SPLIT)"),
    ("A2_vol_doji_is_true_doji", CV2, "return np.where(d == 0, v, np.where(", "return np.where(d == 0, 0, np.where("),
    ("A3_vol0_not_agreed", CV2, "np.where((v == 0) | (v == d), d, AGREED_SPLIT)", "np.where((v == d), d, AGREED_SPLIT)"),
    # --- L: legs ---
    ("L1_split_starts_cont", IP, "    if j > cts_anchor_idx or agreed[j] != sd:",
     "    if j > cts_anchor_idx or agreed[j] == -sd:"),
    ("L2_last_opp_no_doji_skip", IP, '''    while j <= cts_anchor_idx and agreed[j] == 0:
        j += 1''', '''    while False and agreed[j] == 0:
        j += 1'''),
    ("L3_cont_no_dojis", IP, _CONT_RUN, _CONT_RUN.replace("(0, sd)", "(sd,)")),
    ("L4_cont_cap_plus1", IP, "    capped = run[:cap]", "    capped = run[:cap + 1]"),
    ("L5_cont_no_e1", IP, "    cont = _trim(capped, agreed, lead=False)", "    cont = list(capped)"),
    ("L6_confirmed_last", IP, _CONFIRMED, "    confirmed = capped[-1]"),
    ("L7_confirmed_next", IP, _CONFIRMED, "    confirmed = capped[-1] + 1"),
    ("L8_cont_past_cts", IP, _CONT_RUN, _CONT_RUN.replace("j <= cts_anchor_idx", "j < len(agreed)")),
    ("L9_opp_cut_to_ic", IP, '''    while j >= 0 and agreed[j] in (0, -sd):
        opp.insert(0, j)''', '''    while False:
        opp.insert(0, j)'''),
    ("L10_no_junction", IP, "    while len(opp) > 1 and agreed[opp[0]] == 0:", "    while False:"),
    ("L11_bos_exclusive", IP, _INIT_RUN, _INIT_RUN.replace(">=", ">")),
    ("L12_init_no_dojis", IP, _INIT_RUN, _INIT_RUN.replace("(0, sd)", "(sd,)")),
    ("L13_rule_ab_zero", IP, "    rule_ab = min(max(cap - len(full), 0), len(junction))", "    rule_ab = 0"),
    ("L14_init_uncapped", IP, _INITS, "        inits = [(_trim(seq, agreed, lead=True), False)]"),
    ("L15_init_no_e1", IP, _INITS, "        inits = [(list(seq[-cap:]), False)]"),
    ("L16_no_d3c", IP, "            inits.append((seq[-1:], True))", "            pass"),
    ("L17_doji_initial_ok", IP, "            if not any(agreed[x] == sd for x in init):", "            if not init:"),
    ("L18_one_placement", IP, "    for placement in range(len(junction) + 1):", "    for placement in [rule_ab]:"),
    # --- R: ranks / stacks ---
    ("R1_ref_at_ic", IP, "    ref = float(fr.ref[v.initial[0]])", "    ref = float(fr.ref[ic])"),
    ("R2_body_ge", IP, '    if not e["body"] > m["body"]:', '    if not e["body"] >= m["body"]:'),
    ("R3_cont_big_normal_ok", IP, '    if not e["big_maru"]:', '    if not (e["big_maru"] or e["big_normal"]):'),
    ("R4_big_normal_no_type", IP, '"big_normal": typ == "normal" and flags is not None and bool(flags[2]),',
     '"big_normal": flags is not None and bool(flags[2]),'),
    ("R5_big_maru_no_type", IP, '"big_maru": typ == "maru" and flags is not None and bool(flags[1]),',
     '"big_maru": flags is not None and bool(flags[1]),'),
    ("R6_stack_no_floor", IP, "    types = classify_candles(compute_candle_metrics(stacks), fr.params)",
     "    types = classify_candles(compute_candle_metrics(stacks), CandleParams())"),
    ("R7_3c_on_cap1", IP, '        if rank == "3C" and not v.only_3d:', '        if rank == "3C":'),
    ("R8_3d_multi_init", IP, '        if rank == "3D" and not multi_init:', '        if rank == "3D":'),
    ("R9_3d_shape", IP, '("3D", "ii" if multi_cont else "i")', '("3D", _SHAPE[(multi_init, multi_cont)])'),
    ("R10_cap1_any_rank", IP, '''    if v.only_3d:
        return None, "3d_needs_multi_opp", detail''', '''    if False:
        return None, "3d_needs_multi_opp", detail'''),
    ("R11_rank_1A_1B", IP, '("big_maru", "maru"): "1A",', '("big_maru", "maru"): "1B",'),
    ("R12_shape_swap", IP, '(True, False): "ii", (False, True): "iii"', '(True, False): "iii", (False, True): "ii"'),
    ("R13_tie_first", IP, "(key == best[0] and v.placement == rule_ab", "(False and v.placement == rule_ab"),
    ("R14_tie_last", IP, "if best is None or key < best[0] or", "if best is None or key <= best[0] or"),
    ("R15_single_stack_reclassified", IP, '    typ = str(fr.ctype[first]) if first == last else types[(first, last)]',
     '    typ = types[(first, last)] if (first, last) in types else "maru"'),
    # --- S: selection ---
    ("S1_classify_all", POI, "    to_classify = [i for i in candidates if classify_misses or touches[i]]",
     "    to_classify = list(candidates)"),
    ("S2_pick_ignores_touch", POI, "    patterns = [c.ip for c in cands if c.touches_zone and c.ip.rank is not None]",
     "    patterns = [c.ip for c in cands if c.ip.rank is not None]"),
    ("S3_candle_direction", POI,
     "    candidates = find_ic_candidates(df, fib_state, config, ic_direction=pd.Series(agreed, index=df.index))",
     "    candidates = find_ic_candidates(df, fib_state, config)"),
    ("S4_touch_exclusive", POI, "    return high >= zone_lo and low <= zone_hi", "    return high > zone_lo and low < zone_hi"),
    ("S5_tb1_reversed", POI, "    best_type = min(_type_rank(r) for r in tied)", "    best_type = max(_type_rank(r) for r in tied)"),
    ("S6_tb2_earliest", POI, _TB2_PICK, _TB2_PICK.replace("pick = max(", "pick = min(")),
    ("S7_tb2_le", POI, _TB2_PICK, _TB2_PICK.replace("r.ic_idx < reference", "r.ic_idx <= reference")),
    ("S8_tb2_from_newest", POI, "    oldest = min(r.ic_idx for r in tb1)", "    oldest = max(r.ic_idx for r in tb1)"),
    ("S9_tb2_counts_filled", POI, "            df, oldest + 1, fib_state.cts_idx, fib_state.cts_idx, config.fill_threshold,",
     "            df, oldest + 1, fib_state.cts_idx, fib_state.cts_idx, 1e9,"),
    ("S10_tb2_runner_up", POI, "    runner_up = max(r.ic_idx for r in tb1 if r is not pick)",
     "    runner_up = min(r.ic_idx for r in tb1 if r is not pick)"),
    ("S11_order_letter_only", IP, "        return RANKS.index(self.rank), SHAPES.index(self.shape)",
     "        return RANKS.index(self.rank), 0"),
    ("S12_zone_one_level", POI, "    b = calculate_fib_price(anchor_high, anchor_low, config.ic_fib_max, sd)",
     "    b = calculate_fib_price(anchor_high, anchor_low, 100.0, sd)"),
    ("S13_tb1_runner_up", POI, "        runner_up = max(r.ic_idx for r in losers if _type_rank(r) == runner_type)",
     "        runner_up = min(r.ic_idx for r in losers)"),
    # --- G: the record gate / rows ---
    ("G1_gate_ignores_end", POI, "    return bool((fib_state.active and fib_state.end_idx is None) or fib_state.locked)",
     "    return bool(fib_state.active or fib_state.locked)"),
    ("G2_no_est_skip", POI, "        if (sid, cycle_id) not in established:", "        if False:"),
    ("G3_no_none_row", POI, '''        if sel.pick is None:
            out.append(IPCandidateRecord(''', '''        if False:
            out.append(IPCandidateRecord('''),
    ("G4_no_misses", POI, "        sel = select_ip_ic(df, fib_state, config, classify_misses=True)",
     "        sel = select_ip_ic(df, fib_state, config, classify_misses=False)"),
    ("G5_tiebreak_every_row", POI,
     "            tb = sel.tiebreak if picked and sel.tiebreak is not None else (None, None, None, None)",
     "            tb = sel.tiebreak if sel.tiebreak is not None else (None, None, None, None)"),
    # --- M: the sub mirror ---
    ("M1_no_shift", EDM, "            f: int(getattr(r, f)) + slice_begin", "            f: int(getattr(r, f))"),
    ("M2_list_missing", EDM, '''    "pattern_confirmed_idx",
    "ip_tb_runner_up_ic_idx",''', '''    "ip_tb_runner_up_ic_idx",'''),
    ("M3_no_attribution", EDM, "        new_ip.append(replace(r, meta={**r.meta, **attribution}, **shifted))",
     "        new_ip.append(replace(r, **shifted))"),
    ("M4_projection_drops", EDM, '        ip_candidates=down["ip_candidates"],', "        ip_candidates=[],"),
    # --- O / X / C / Q / I / B: wiring ---
    ("O1_downstream_empty", ORCH, '        "ip_candidates": ip_candidates,', '        "ip_candidates": [],'),
    ("O2_config_literals_back", ORCH, "    poi_config = POIConfig()", "    poi_config = POIConfig(fill_threshold=0.70, ip_leg_cap_by_tf={'H1': 2, 'M15': 2})"),
    ("X1_legs_dropped", EXP, '        row["legs"] = _ip_legs(r.ic_idx, r.ip_leg_candle_counts)', '        row["legs"] = None'),
    ("X2_legs_offset", EXP, '    opp_first = ic_idx - n_opp + 1', '    opp_first = ic_idx - n_opp'),
    ("C1_stamp_constant", CC, '    df.attrs["timeframe"] = str(timeframe)', '    df.attrs["timeframe"] = "H1"'),
    ("Q1_qualified_vol_strict", WC, "    return int(agreed_direction(direction, vol_dir)) == required_dir",
     "    return vol_dir == required_dir"),
    ("I1_iter_yields_filled", IMB, '''        if not inst.is_filled(df, check_to_idx, fill_threshold):
            yield inst''', '''        if True:
            yield inst'''),
    ("B1_ratio_unrounded", CV2, "    ratio = round(length / prior_max, 2)", "    ratio = length / prior_max"),
    # --- the landing review's test-gap lens (2026-10-06) ---
    # --- legs ---
    ("N01_rule_ab_all_junction", IP, "    rule_ab = min(max(cap - len(full), 0), len(junction))",
     "    rule_ab = len(junction) if len(full) < cap else 0"),
    ("N02_d3c_from_full", IP, "            inits.append((seq[-1:], True))", "            inits.append((full[-1:], True))"),
    ("N03_cont_trim_wrong_end", IP, "    cont = _trim(capped, agreed, lead=False)", "    cont = _trim(capped, agreed, lead=True)"),
    ("N05_confirmed_run_gt_cap", IP, "    confirmed = capped[-1] if len(capped) == cap else capped[-1] + 1",
     "    confirmed = capped[-1] if len(run) > cap else capped[-1] + 1"),
    # --- stacks / ranks ---
    ("N06_ref_as1", IP, '        self.ref = df["prior_maru_max_len_as0"].to_numpy(dtype=float)',
     '        self.ref = df["prior_maru_max_len_as1"].to_numpy(dtype=float)'),
    ("N07_ref_initial_last", IP, "    ref = float(fr.ref[v.initial[0]])", "    ref = float(fr.ref[v.initial[-1]])"),
    ("N08_body_last_candle", IP, '        "body": abs(c - o),', '        "body": abs(fr.c[last] - fr.o[last]),'),
    ("N10_stack_high_last", IP, "float(self.h[first:last + 1].max())", "float(self.h[last])"),
    ("N11_stack_low_first", IP, "float(self.l[first:last + 1].min())", "float(self.l[first])"),
    ("N12_ratio_on_body", IP, "    flags = big_ratio_flags(h - l, ref, fr.params)",
     "    flags = big_ratio_flags(abs(c - o), ref, fr.params)"),
    ("N13_stack_no_pair", IP, '    stacks.attrs["pair"] = fr.pair\n', ""),
    ("N14_first_fail_last", IP, "                first_fail = first_fail or (why, v, detail)",
     "                first_fail = (why, v, detail)"),
    # --- selection ---
    ("N15_tb1_over_patterns", POI, "    tb1 = [r for r in tied if _type_rank(r) == best_type]",
     "    tb1 = [r for r in patterns if _type_rank(r) == best_type]"),
    ("N16_tb2_evaluated_at_cts", POI, "direction=sd, evaluated_at=None,", "direction=sd, evaluated_at=fib_state.cts_idx,"),
    ("N17_tb2_horizon_frame_end", POI,
     "            df, oldest + 1, fib_state.cts_idx, fib_state.cts_idx, config.fill_threshold,",
     "            df, oldest + 1, fib_state.cts_idx, len(df) - 1, config.fill_threshold,"),
    ("N19_tb2_count_tied", POI, '("TB2", int(runner_up), len(tb1), int(reference.start_idx))',
     '("TB2", int(runner_up), len(tied), int(reference.start_idx))'),
    ("N32_select_params_h1", POI, "    params = candle_params_for_timeframe(timeframe)",
     '    params = candle_params_for_timeframe("H1")'),
    ("N33_zone_sd_const", POI,
     "    a = calculate_fib_price(anchor_high, anchor_low, config.ic_fib_min, sd)\n"
     "    b = calculate_fib_price(anchor_high, anchor_low, config.ic_fib_max, sd)",
     "    a = calculate_fib_price(anchor_high, anchor_low, config.ic_fib_min, 1)\n"
     "    b = calculate_fib_price(anchor_high, anchor_low, config.ic_fib_max, 1)"),
    # --- records ---
    ("N20_none_reason_const", POI, 'reason="no_pattern" if sel.candidates else "no_candidates",', 'reason="no_pattern",'),
    ("N21_ip_gate_no_locked", POI,
     "        if not _poi_record_gate(fib_state):\n            continue\n        sid, cycle_id = int(",
     "        if not (fib_state.active and fib_state.end_idx is None):\n            continue\n        sid, cycle_id = int("),
    ("N22_dedupe_records", POI, "    for fib_state in fib_tracker.get_fibs_for_charting():",
     "    for fib_state in {(f.structure_id, f.cycle_id): f for f in fib_tracker.get_fibs_for_charting()}.values():"),
    ("N23_base_bos_cts_swap", POI,
     "int(fib_state.struct_direction), int(fib_state.bos_idx), int(fib_state.cts_idx))",
     "int(fib_state.struct_direction), int(fib_state.cts_idx), int(fib_state.bos_idx))"),
    ("N24_rec_anchor_opp_swap", POI,
     "                pattern_anchor_idx=r.pattern_anchor_idx,\n                opp_leg_first_idx=r.opp_leg_first_idx,",
     "                pattern_anchor_idx=r.opp_leg_first_idx,\n                opp_leg_first_idx=r.pattern_anchor_idx,"),
    ("N25_rec_agreed_is_direction", POI,
     'agreed_dir=int(agreed_direction(df.loc[r.ic_idx, "direction"], df.loc[r.ic_idx, "vol_dir"])),',
     'agreed_dir=int(df.loc[r.ic_idx, "direction"]),'),
    # --- export / mirror ---
    ("N26_exp_touch_is_picked", EXP, '            "touches_zone": r.touches_zone,', '            "touches_zone": r.picked,'),
    ("N27_exp_picked_is_touch", EXP, '            "picked": r.picked,', '            "picked": r.touches_zone,'),
    ("N28_exp_legs_cont_ignores_count", EXP, "|{ic_idx + 1}..{ic_idx + n_cont}\"", "|{ic_idx + 1}..{ic_idx + 1}\""),
    ("N30_exp_no_object_dtype", EXP, "pd.DataFrame(rows, columns=IP_CANDIDATE_COLUMNS, dtype=object)",
     "pd.DataFrame(rows, columns=IP_CANDIDATE_COLUMNS)"),
    ("N31_mirror_overwrites", EDM, '    _attrs_setdefault_list(entity_df, "ip_candidates").extend(new_ip)',
     '    entity_df.attrs["ip_candidates"] = new_ip'),
    # --- the landing review's code lens (2026-10-06): the fail-loud reads ---
    ("F1_index_check_loose", IP, "    assert df.index.equals(pd.RangeIndex(len(df))), (",
     "    assert len(df) == 0 or int(df.index[-1]) == len(df) - 1, ("),
    ("F2_pair_default", IP, '        self.pair = df.attrs["pair"]', '        self.pair = df.attrs.get("pair", "")'),
    ("F3_locked_assert_dropped", POI,
     '            assert not fib_state.locked, f"[poi_zones] locked fib {(sid, cycle_id)} has no CTS_ESTABLISHED"',
     "            pass"),
]
