"""PLAN_K landing A's review fix-up (§11.2) — mutants for `mutant_runner.py` (full suite each).

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \
        engine_v2/plans/plan_h_inputs/mutants_k_landing_a_review.py --jobs 13

Lens B's 9 survivors on `02e9381` (N11 re-expressed: `ic_reference` no longer takes `end_idx`, so the rejected
cut-at-the-moment reading moves into the frame handed in) + the LA1 fix (the final reference = as of the lifecycle
cap, the watch on the whole frame). Every one must be KILLED.
"""
WATCH = "engine_v2/patterns/cross_limit_watch.py"
PZ = "engine_v2/zones/poi_zones.py"
FT = "engine_v2/zones/fib_tracker.py"
ZR = "engine_v2/charting/_zone_render.py"
CFL = "engine_v2/multitf/cross_fib_limit.py"
SL = "engine_v2/zones/structure_lifecycle.py"

MUT = [
    # --- lens B's survivors (each pinned by tests/test_plan_k_landing_a_review_pins.py) ---
    ("N04_early_stop_too_early", WATCH,
     "if best is not None and a >= best[1]:", "if best is not None and a >= best[1] - 2:"),
    ("N10_reason_touching_only", PZ,
     "ref_reasons = {i: ic_reference_reason(df, i, ic_reference, sd) for i in candidates}",
     "ref_reasons = {i: (ic_reference_reason(df, i, ic_reference, sd) if touches[i] else None) for i in candidates}"),
    ("N11_ms_frame_cut_at_moment", PZ,
     "ref = tracker.ic_reference(fib_state, df, evaluated_at=int(evaluated_at))",
     "ref = tracker.ic_reference(fib_state, df.iloc[:int(evaluated_at) + 1], evaluated_at=int(evaluated_at))"),
    ("N17_cycle_on_structure_ref", ZR,
     '    if source == "prev_cycle_cts":\n', '    if source != "cross_limit_anchor":\n'),
    ("N23_rev_first_bos", CFL,
     "    last_bos = max(boses, key=ef.event_moment)\n    sb = int(slice_begin)",
     "    last_bos = boses[0]\n    sb = int(slice_begin)"),
    ("N24_rev_bos_strict", CFL,
     "ef.event_moment(e) <= int(sc[0].idx)]", "ef.event_moment(e) < int(sc[0].idx)]"),
    ("N25_rev_last_in_list", CFL,
     "    last_bos = max(boses, key=ef.event_moment)\n    sb = int(slice_begin)",
     "    last_bos = boses[-1]\n    sb = int(slice_begin)"),
    ("N27_tripwire_ignores_break", CFL,
     "        if ((got.price, got.source, got_from, got.ref_anchor_price, got.break_pattern_anchor_idx,\n"
     "             got.break_confirmed_idx)",
     "        if ((got.price, got.source, got_from, got.ref_anchor_price, stored.break_pattern_anchor_idx,\n"
     "             stored.break_confirmed_idx)"),
    ("N28_main_bos_strict", SL,
     "             and ef.event_moment(ev) <= int(rev.idx)]", "             and ef.event_moment(ev) < int(rev.idx)]"),
    # --- the LA1 fix ---
    ("F1_watch_frame_cut_at_moment", FT,
     "brk = find_cross_limit_break(df, float(lim.price), sd, start, known_at=evaluated_at)",
     "brk = find_cross_limit_break(df, float(lim.price), sd, start, end_idx=evaluated_at, known_at=evaluated_at)"),
    ("F2_stamp_retrospective", FT,
     "ref = self.ic_reference(fib, df, evaluated_at=lifecycle_cap)",
     "ref = self.ic_reference(fib, df, evaluated_at=None)"),
    ("F3_drawn_retrospective", PZ,
     "ic_reference=fib_tracker.ic_reference(fib_state, df, evaluated_at=lifecycle_cap),",
     "ic_reference=fib_tracker.ic_reference(fib_state, df, evaluated_at=None),"),
    ("F4_watch_ignores_the_moment", FT,
     "brk = find_cross_limit_break(df, float(lim.price), sd, start, known_at=evaluated_at)",
     "brk = find_cross_limit_break(df, float(lim.price), sd, start, known_at=None)"),
]
