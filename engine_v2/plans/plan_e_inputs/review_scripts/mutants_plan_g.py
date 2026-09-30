"""Mutants of Plan G (WVMI on the unique sub, `plans/PLAN_G_wvmi_unique_sub.md`) for `mutant_runner.py`.

The cold review's 28 (`plans/plan_g_inputs/cold_review_20260930.md`: G1a-e, G2a-f, G3a-d, G4a-i, X1-3, M1; G3d "the
persister kept" is moot — deleted — so G3d here is its row-count equivalent, the mirror persisting twice) + the frame
(G1f / G1g), the capped-record assert (G1i), Q10's lock fallback (W1 / W2), the call-site swaps / counts / parent path
(G4b2 / G4i / G4l), the gate aliasing / key order (G1j / M2), proximity for subs (M3), the mode validation (M4), the
sort (G4j / G4k), the merge order (G4c2), the counts (C1 / C2) and the exporter's Int64 (X4). Value / swap / extra-key
/ ordering / aliasing / exporter shapes, not only reverts.
Template: `(id, relpath, old, new)`; `old` must occur exactly once in the file.
"""
ORCH = "engine_v2/pipeline/orchestrator.py"
PSB = "engine_v2/multitf/pooled_structure_build.py"
EDM = "engine_v2/multitf/entity_df_mutation.py"
TYP = "engine_v2/common/types.py"
EXP = "engine_v2/debug/export_wvmi.py"
WV = "engine_v2/zones/wvmi.py"

_ENDS = "cycle_end_by_key = {key: end for key, (_start, end, _reason) in life.items() if end is not None}"
_TRK = "tracker = WVMITracker(structure_path_id=structure_path_id, cycle_end_by_key=cycle_end_by_key)"
_FRAME = "frame = df.iloc[:lifecycle_cap + 1] if lifecycle_cap is not None else df"
_FLAG = "rec.cycle_collapsed = end is not None and start >= end"
_GATE = '''        if trigs and trigs[0].direction == "sd":
            first_sd = trigs[0]'''
_HIT = "hit = next((t for t in mapped if start <= t[0] <= end), None)"
_MERGE = "rec.meta = {**trigger, **{k: v for k, v in rec.meta.items() if k not in trigger}}"
_JOIN = '''        for rec in lens_df.attrs.get("wvmi", []):
            trigger = trigger_by_sub[rec.meta["sub_id"]]'''
_COPY = '''    for w in result.wvmi_records:
        nw = deepcopy(w)'''
_LOCK = '''        if lp_idx is not None and lp_idx in df.index:
            rec.lp_idx = lp_idx'''

MUT = [
    # --- G1: the mode, the ends, the frame, the assert ---
    ("G1a_default_none", ORCH, '    wvmi: str = "first_sd_prox",', '    wvmi: str = "none",'),
    ("G1b_projection_gated", PSB, 'wvmi="none",', 'wvmi="first_sd_prox",'),
    ("G1c_projection_off", PSB, 'wvmi="none",', 'wvmi="off",'),
    ("G1d_no_ends_for_subs", ORCH, _TRK,
     "tracker = WVMITracker(structure_path_id=structure_path_id, "
     "cycle_end_by_key=cycle_end_by_key if gate is not None else {})"),
    ("G1e_ends_plus_one", ORCH, _ENDS, _ENDS.replace("{key: end for", "{key: end + 1 for")),
    ("G1f_no_frame", ORCH, _FRAME, "frame = df"),
    ("G1g_frame_off_by_one", ORCH, _FRAME, "frame = df.iloc[:lifecycle_cap] if lifecycle_cap is not None else df"),
    ("G1h_lock_on_natural_frame", ORCH, "tracker.on_bos_confirmed(ev, frame, wave_candle_results)",
     "tracker.on_bos_confirmed(ev, df, wave_candle_results)"),
    ("G1i_assert_dropped", ORCH, "assert lifecycle_cap is None or key in cycle_end_by_key, (",
     "assert True or key in cycle_end_by_key, ("),
    ("G1j_gate_meta_aliased", ORCH, "rec.meta.update(gate[key])", "rec.meta = gate[key]"),
    # --- G2: cycle_collapsed ---
    ("G2a_gt", ORCH, _FLAG, "rec.cycle_collapsed = end is not None and start > end"),
    ("G2b_eq", ORCH, _FLAG, "rec.cycle_collapsed = end is not None and start == end"),
    ("G2c_next_cycle_key", ORCH, "start, end, _reason = life[key]",
     "start, end, _reason = life.get((key[0], key[1] + 1), life[key])"),
    ("G2d_subs_only", ORCH, _FLAG, "rec.cycle_collapsed = gate is None and end is not None and start >= end"),
    ("G2e_open_flagged", ORCH, _FLAG, "rec.cycle_collapsed = end is None or start >= end"),
    ("G2f_default_true", TYP, "    cycle_collapsed: bool = False", "    cycle_collapsed: bool = True"),
    # --- G3: the mirror ---
    ("G3a_field_not_set", EDM, "        nw.structure_path_id = structure_path_id\n", ""),
    ("G3b_field_on_source", EDM, _COPY,
     "    for w in result.wvmi_records:\n        w.structure_path_id = structure_path_id\n        nw = deepcopy(w)"),
    ("G3c_shallow_copy", EDM, _COPY, "    for w in result.wvmi_records:\n        nw = replace(w)"),
    ("G3d_persisted_twice", EDM, '_attrs_setdefault_list(entity_df, "wvmi").extend(new_wvmis)',
     '_attrs_setdefault_list(entity_df, "wvmi").extend(new_wvmis + deepcopy(new_wvmis))'),
    # --- G4: the per-lens trigger post-pass, the stream mapping, the call site ---
    ("G4a_streams_unioned", ORCH, "        for parent_idx, event_type in streams_by_lens[lens]:",
     "        for parent_idx, event_type in [e for _l in streams_by_lens for e in streams_by_lens[_l]]:"),
    ("G4b_mapping_swapped", ORCH, "    return {LENS_CONFLUENCE: conf_stream, LENS_COUNTER: ctr_stream}",
     "    return {LENS_CONFLUENCE: ctr_stream, LENS_COUNTER: conf_stream}"),
    ("G4b2_call_site_var_swap", ORCH, "tables.cycles(), main_zone_proximity_triggers, var3_all, var4_all,",
     "tables.cycles(), main_zone_proximity_triggers, var4_all, var3_all,"),
    ("G4c_update_not_prepend", ORCH, _MERGE, "rec.meta.update(trigger)"),
    ("G4c2_meta_wins", ORCH, _MERGE, "rec.meta = {**trigger, **rec.meta}"),
    ("G4d_extra_m15_key", ORCH, '                "parent_path_id": parent_path_id,\n',
     '                "parent_path_id": parent_path_id,\n'
     '                "triggered_by_event_m15_idx": hit[0] if hit is not None else None,\n'),
    ("G4e_loh_idx_stored", ORCH, '"triggered_by_event_idx": hit[1] if hit is not None else None,',
     '"triggered_by_event_idx": hit[0] if hit is not None else None,'),
    ("G4f_parent_path_none", ORCH, '                "parent_path_id": parent_path_id,\n',
     '                "parent_path_id": parent_path_id if hit is not None else None,\n'),
    ("G4g_end_exclusive", ORCH, _HIT, "hit = next((t for t in mapped if start <= t[0] < end), None)"),
    ("G4g2_start_exclusive", ORCH, _HIT, "hit = next((t for t in mapped if start < t[0] <= end), None)"),
    ("G4h_join_by_position", ORCH, _JOIN,
     '        for _i, rec in enumerate(lens_df.attrs.get("wvmi", [])):\n'
     '            trigger = list(trigger_by_sub.values())[min(_i, len(trigger_by_sub) - 1)]'),
    ("G4i_counted_per_lens", ORCH, "wvmi_counts = _count_sub_wvmi(all_results)",
     "wvmi_counts = _count_sub_wvmi(results_by_lens[LENS_CONFLUENCE] + results_by_lens[LENS_COUNTER])"),
    ("G4j_tie_reversed", ORCH, "mapped.sort(key=lambda t: (t[0], t[1]))", "mapped.sort(key=lambda t: (t[0], -t[1]))"),
    ("G4k_unsorted", ORCH, "mapped.sort(key=lambda t: (t[0], t[1]))", "pass"),
    ("G4l_parent_path_from_lens", ORCH, "lens_dfs=lens_dfs, parent_path_id=parent_path,",
     "lens_dfs=lens_dfs, parent_path_id=lens_paths[LENS_CONFLUENCE],"),
    # --- the counts ---
    ("C1_by_lens_set_order", ORCH, 'for l in sorted(set(res.meta.get("lenses") or ())):',
     'for l in set(res.meta.get("lenses") or ()):'),
    ("C2_empty_sub_acted", ORCH, "        if not recs:\n            continue\n", ""),
    # --- the exporter ---
    ("X1_column_reads_lp_locked", EXP, '"cycle_collapsed": r.cycle_collapsed,', '"cycle_collapsed": r.lp_locked,'),
    ("X2_column_reads_meta", EXP, '"cycle_collapsed": r.cycle_collapsed,', '"cycle_collapsed": (r.meta or {}).get("cycle_collapsed"),'),
    ("X3_column_order", EXP, '    "lp_locked",\n    "cycle_collapsed",', '    "cycle_collapsed",\n    "lp_locked",'),
    ("X4_no_int64", EXP, '    out["triggered_by_event_idx"] = out["triggered_by_event_idx"].astype("Int64")\n', ""),
    # --- the main gate, proximity, the mode ---
    ("M1_any_sd", ORCH, _GATE,
     '        if any(t.direction == "sd" for t in trigs):\n'
     '            first_sd = next(t for t in trigs if t.direction == "sd")'),
    ("M2_gate_key_order", ORCH, "rec.meta.update(gate[key])",
     "rec.meta.update(dict(reversed(list(gate[key].items()))))"),
    ("M3_proximity_for_subs", ORCH, '    if wvmi == "first_sd_prox":\n        # Zone proximity triggers',
     '    if wvmi != "off":\n        # Zone proximity triggers'),
    ("M4_no_validation", ORCH, "    if wvmi not in WVMI_MODES:\n        raise", "    if False:\n        raise"),
    # --- Q10: the lock fallback ---
    ("W1_lock_reads_outside", WV, _LOCK, "        if lp_idx is not None:\n            rec.lp_idx = lp_idx"),
    ("W2_old_lock_behaviour", WV, _LOCK,
     "        if lp_idx is not None and lp_idx not in df.index:\n"
     "            rec.lp_idx, rec.lp_volume, rec.lp_weight, rec.pullback_momentum = lp_idx, None, 1.0, None\n"
     "        elif lp_idx is not None:\n"
     "            rec.lp_idx = lp_idx"),
]
