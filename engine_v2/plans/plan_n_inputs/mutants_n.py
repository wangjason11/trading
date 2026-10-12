"""PLAN_N (the Naming-Standard audit) — mutants for `mutant_runner.py` (full suite each; PLAN_N §5's list + every
helper call site).

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \
        engine_v2/plans/plan_n_inputs/mutants_n.py --jobs 14 --out <a short %TEMP% dir>

Every one must be KILLED. Declared unpinnable (not listed): the KL step sort keys (steps are appended in index order —
39 / 39 sorted, no ties), the three renderer fallback step builders (unreachable: every zone has an INIT step), and
the first two passes of the H1 range hover-line builder (they build state nothing reads).
"""
MS = "engine_v2/structure/market_structure.py"
KL = "engine_v2/zones/kl_zones_v1.py"
ZR = "engine_v2/charting/_zone_render.py"
EP = "engine_v2/charting/export_plotly.py"
EM = "engine_v2/charting/export_m15_chart.py"
EDM = "engine_v2/multitf/entity_df_mutation.py"
SW = "engine_v2/multitf/lifecycle_sweep.py"
PT = "engine_v2/multitf/parent_tables.py"

_OLD_GET = 'int(s.get("start_idx", -1))'
_OLD_NEXT = 'int(steps[k + 1].get("start_idx", -1))'

MUT = [
    # --- N3: effective_idx ---
    ("N3a_assert_dropped", MS,
     '        assert "effective_idx" not in m, f"[market_structure] effective_idx is always the state-change idx ({i})"\n',
     ""),
    ("N3b_range_start_restored", MS,
     '        m["effective_idx"] = i\n',
     '        m["effective_idx"] = int(st.range_start_idx) if m.get("reason") == "range_confirmed" else i\n'),
    # --- N2 / N5: RANGE_STARTED ---
    ("N2a_label_anchor_is_the_confirm", MS,
     '"range_anchor_idx": int(st.range_start_idx),   # MS\'s range start (= i; PLAN_N N2)',
     '"range_anchor_idx": int(confirm_idx),'),
    ("N2b_label_confirmed_at_is_the_start", MS,
     '"confirmed_at": int(confirm_idx),              # the moment (== ev.idx)',
     '"confirmed_at": int(i),'),
    ("N5a_pullback_anchor_is_the_apply", MS,
     '"range_anchor_idx": int(st.range_start_idx),\n                        "confirmed_at": int(i),',
     '"range_anchor_idx": int(i),\n                        "confirmed_at": int(i),'),
    ("N5b_proximity_anchor_is_the_candle", MS,
     '"range_anchor_idx": int(st.range_start_idx),\n                        "confirmed_at": int(candle_idx),',
     '"range_anchor_idx": int(candle_idx),\n                        "confirmed_at": int(candle_idx),'),
    ("N2c_registry_drops_range_anchor", EDM,
     '    "range_anchor_idx",     # RANGE_STARTED',
     '    # "range_anchor_idx",     # RANGE_STARTED'),
    # --- N2 / N5: the H1 range chart ---
    ("C1_sort_at_the_event_idx", EP,
     "            key = _range_left_idx(ev)\n",
     "            key = idx\n"),
    ("C2_left_edge_is_the_event_idx", EP,
     '    return int(ev.meta["range_anchor_idx"])',
     "    return int(ev.idx)"),
    ("C3_no_reset_guard", EP,
     "                key = max(key, r)\n",
     "                key = key\n"),
    ("C4_box_site_reads_the_event_idx", EP,
     '                    "start_idx": _range_left_idx(ev),',
     '                    "start_idx": int(ev.idx),'),
    ("C5_hover_pass3_reads_the_event_idx", EP,
     "                start_idx = _range_left_idx(ev)\n                open_ranges[sid] = {\n                    \"start_idx\": start_idx,\n                    \"hi\": float(ev.meta.get(\"hi\", 0)),\n                    \"lo\": float(ev.meta.get(\"lo\", 0)),\n                    \"last_update_idx\": start_idx,\n                    \"struct_direction\": sd,\n                }\n            elif",
     "                start_idx = int(ev.idx)\n                open_ranges[sid] = {\n                    \"start_idx\": start_idx,\n                    \"hi\": float(ev.meta.get(\"hi\", 0)),\n                    \"lo\": float(ev.meta.get(\"lo\", 0)),\n                    \"last_update_idx\": start_idx,\n                    \"struct_direction\": sd,\n                }\n            elif"),
    # --- N1: KL steps (writers, helper, mirror) ---
    ("K1_init_writes_the_confirm", KL,
     '"base_idx": int(base_idx),   # the segment begins at the base pattern\'s first candle',
     '"base_idx": int(confirmed_idx),'),
    ("K2_expansion_writes_the_anchor", KL,
     '"expanded_idx": ef.event_moment(ev),  # the expansion\'s MOMENT (the update\'s processing candle)',
     '"expanded_idx": int(z0.meta["anchor_idx"]),'),
    ("K3_helper_keys_swapped", KL,
     '    return "base_idx" if step["event"] == "INIT" else "expanded_idx"',
     '    return "expanded_idx" if step["event"] == "INIT" else "base_idx"'),
    ("K4_mirror_shifts_only_init", EDM,
     "                new_step[k] = int(new_step[k]) + slice_begin\n",
     "                new_step[k] = int(new_step[k]) + (slice_begin if k == \"base_idx\" else 0)\n"),
    # --- N1: each renderer read left on the old key (the realistic miss) ---
    ("R01_outline_step_x0", ZR,
     "        step_x0 = x_resolver(bounds_step_draw_idx(s))",
     f"        step_x0 = x_resolver({_OLD_GET})"),
    ("R02_outline_next_x0", ZR,
     "            nxt_x0 = x_resolver(bounds_step_draw_idx(sorted_steps[k + 1]))",
     '            nxt_x0 = x_resolver(int(sorted_steps[k + 1].get("start_idx", -1)))'),
    ("R03_h1_seg_start", EP,
     "                seg_start_idx = bounds_step_draw_idx(s)",
     f"                seg_start_idx = {_OLD_GET}"),
    ("R04_h1_next_idx", EP,
     "                    next_idx = bounds_step_draw_idx(steps[k + 1])",
     f"                    next_idx = {_OLD_NEXT}"),
    ("R05_h1_confirm_seg_x0", EP,
     "                    seg_x0 = _idx_to_time(bounds_step_draw_idx(s))",
     f"                    seg_x0 = _idx_to_time({_OLD_GET})"),
    ("R06_h1_confirm_next", EP,
     "                        nxt = _idx_to_time(bounds_step_draw_idx(steps[k + 1])) or x_zone1",
     f"                        nxt = _idx_to_time({_OLD_NEXT}) or x_zone1"),
    ("R07_m15sub_seg_start", EM,
     "                    seg_start_slice = bounds_step_draw_idx(s)",
     f"                    seg_start_slice = {_OLD_GET}"),
    ("R08_m15sub_next_idx", EM,
     "                        next_step_idx = bounds_step_draw_idx(steps[k + 1])",
     f"                        next_step_idx = {_OLD_NEXT}"),
    ("R09_m15sub_confirm_seg_x0", EM,
     "                        seg_x0 = _lt_time(bounds_step_draw_idx(s))",
     f"                        seg_x0 = _lt_time({_OLD_GET})"),
    ("R10_m15sub_confirm_next", EM,
     "                            nxt = _lt_time(bounds_step_draw_idx(steps[k + 1])) or x1",
     f"                            nxt = _lt_time({_OLD_NEXT}) or x1"),
    ("R11_m15h1_seg_start", EM,
     "                seg_start_h1 = bounds_step_draw_idx(s)",
     f"                seg_start_h1 = {_OLD_GET}"),
    ("R12_m15h1_next_idx", EM,
     "                    next_h1 = bounds_step_draw_idx(steps[k + 1])",
     f"                    next_h1 = {_OLD_NEXT}"),
    ("R13_m15h1_confirm_seg_x0", EM,
     "                    seg_x0 = _h1_idx_to_m15_time(bounds_step_draw_idx(s))",
     f"                    seg_x0 = _h1_idx_to_m15_time({_OLD_GET})"),
    ("R14_m15h1_confirm_next", EM,
     "                        nxt = _h1_idx_to_m15_time(bounds_step_draw_idx(steps[k + 1])) or x1",
     f"                        nxt = _h1_idx_to_m15_time({_OLD_NEXT}) or x1"),
    # --- N4: collapsed ---
    ("D1_sweep_reason_reverted", SW,
     '                t, "collapsed_parent_cycle",',
     '                t, "degenerate_parent_cycle",'),
    ("D2_warning_text_reverted", PT,
     '                f"WARNING [parent_tables] collapsed parent cycle {key}: "',
     '                f"WARNING [parent_tables] degenerate parent cycle {key}: "'),
]
