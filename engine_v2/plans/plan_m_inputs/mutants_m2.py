"""PLAN_M landing 2 (the counter-sub source) — mutants for `mutant_runner.py` (full suite each; PLAN_M §9 L2's list +
the call sites).

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \
        engine_v2/plans/plan_m_inputs/mutants_m2.py --jobs 14 --out <a short %TEMP% dir>

Every one must be KILLED.
"""
CFL = "engine_v2/multitf/cross_fib_limit.py"
PT = "engine_v2/multitf/parent_tables.py"
ORC = "engine_v2/pipeline/orchestrator.py"
SW = "engine_v2/multitf/lifecycle_sweep.py"

MUT = [
    # the counter test
    ("N01_counter_test_inverted", CFL,
     "    if int(direction) != int(parent_tables.parent_sd[int(S)]):",
     "    if int(direction) == int(parent_tables.parent_sd[int(S)]):"),
    ("N02_counter_skipped_for_reversal", CFL,
     "    if int(direction) != int(parent_tables.parent_sd[int(S)]):",
     "    if trigger_type != \"reversal\" and int(direction) != int(parent_tables.parent_sd[int(S)]):"),
    ("N03_counter_reads_sid_0", CFL,
     "    if int(direction) != int(parent_tables.parent_sd[int(S)]):",
     "    if int(direction) != int(parent_tables.parent_sd[0]):"),
    ("N04_counter_after_the_reversal_rule", CFL,
     "    S, C = parent_cycle\n    if int(direction) != int(parent_tables.parent_sd[int(S)]):\n"
     "        return parent_bos_limit(parent_tables, int(S), int(C), int(direction), int(s))\n"
     "    if trigger_type == \"reversal\":\n"
     "        assert reversing_sub is not None, \"[cross_fib_limit] a reversal needs its reversing sub\"\n"
     "        return reversal_limit(reversing_sub)\n",
     "    S, C = parent_cycle\n"
     "    if trigger_type == \"reversal\":\n"
     "        assert reversing_sub is not None, \"[cross_fib_limit] a reversal needs its reversing sub\"\n"
     "        return reversal_limit(reversing_sub)\n"
     "    if int(direction) != int(parent_tables.parent_sd[int(S)]):\n"
     "        return parent_bos_limit(parent_tables, int(S), int(C), int(direction), int(s))\n"),
    # parent_bos_limit: the start, the moves, the collapse, the anchor, the key
    ("N05_start_exclusive", CFL,
     "    start = [p for m, p in collapsed if m <= int(s)][-1]\n    moves = tuple((m, p) for m, p in collapsed if m > int(s))",
     "    start = [p for m, p in collapsed if m < int(s)][-1]\n    moves = tuple((m, p) for m, p in collapsed if m >= int(s))"),
    ("N06_start_is_the_bos_price", CFL,
     "    start = [p for m, p in collapsed if m <= int(s)][-1]",
     "    start = float(steps[0][2])"),
    ("N07_no_collapse", CFL,
     "        if collapsed and collapsed[-1][0] == int(m15):",
     "        if False:"),
    ("N08_collapse_keeps_the_first", CFL,
     "            collapsed[-1] = (int(m15), float(price))",
     "            pass"),
    ("N09_anchor_is_p", CFL,
     "    bos_price = float(steps[0][2])",
     "    bos_price = float(steps[-1][2])"),
    ("N10_key_ignores_the_cycle", CFL,
     "    key = (int(parent_sid), int(parent_cycle_id))\n    steps = tables.bos_threshold_steps.get(key, ())",
     "    key = (int(parent_sid), int(parent_cycle_id))\n    steps = tables.bos_threshold_steps.get((int(parent_sid), 0), ())"),
    ("N11_key_ignores_the_sid", CFL,
     "    key = (int(parent_sid), int(parent_cycle_id))\n    steps = tables.bos_threshold_steps.get(key, ())",
     "    key = (int(parent_sid), int(parent_cycle_id))\n    steps = tables.bos_threshold_steps.get((0, int(parent_cycle_id)), ())"),
    ("N12_moves_from_the_raw_steps", CFL,
     "    moves = tuple((m, p) for m, p in collapsed if m > int(s))",
     "    moves = tuple((int(m), float(p)) for _h, m, p, _e in steps if int(m) > int(s))"),
    # the asserts
    ("N13_no_bos_by_start_assert", CFL,
     "    assert int(steps[0][1]) <= int(s), (", "    assert True or int(steps[0][1]) <= int(s), ("),
    ("N14_no_further_assert", CFL,
     "        assert beyond(cur[2], prev[2], int(direction)), (",
     "        assert True or beyond(cur[2], prev[2], int(direction)), ("),
    ("N15_further_not_strict", CFL,
     "        assert beyond(cur[2], prev[2], int(direction)), (",
     "        assert beyond(cur[2], prev[2], int(direction)) or cur[2] == prev[2], ("),
    ("N16_no_end_assert", CFL,
     "    assert end_h1 is None or all(int(st[0]) < int(end_h1) for st in steps), (",
     "    assert True or end_h1 is None or all(int(st[0]) < int(end_h1) for st in steps), ("),
    ("N17_end_assert_inclusive", CFL,
     "    assert end_h1 is None or all(int(st[0]) < int(end_h1) for st in steps), (",
     "    assert end_h1 is None or all(int(st[0]) <= int(end_h1) for st in steps), ("),
    ("N18_one_bos_assert_dropped", CFL,
     "    assert n_bos == 1 and steps[0][3] == \"BOS_CONFIRMED\", (",
     "    assert n_bos >= 1, ("),
    # the tripwire
    ("N19_tripwire_ignores_the_moves", CFL,
     "             got.break_confirmed_idx, got.price_at_start, got.price_moves)",
     "             got.break_confirmed_idx, got.price_at_start, stored.price_moves)"),
    ("N20_tripwire_ignores_the_start", CFL,
     "             got.break_confirmed_idx, got.price_at_start, got.price_moves)",
     "             got.break_confirmed_idx, stored.price_at_start, got.price_moves)"),
    # the table
    ("N21_steps_first_m15_of_the_hour", PT,
     "                    (int(h1_at), None if m15_at is None else int(m15_at),",
     "                    (int(h1_at), None if m15_at is None else int(m15_at) - 3,"),
    ("N22_steps_bos_only", PT,
     "        if et in (\"BOS_CONFIRMED\", \"BOS_THRESHOLD_UPDATED\"):",
     "        if et in (\"BOS_CONFIRMED\",):"),
    ("N23_steps_keyed_by_sid", PT,
     "                steps.setdefault((int(sid), int(cyc)), []).append(",
     "                steps.setdefault((int(sid), 0), []).append("),
    # the call sites
    ("N24_driver_no_tables_to_the_tripwire", ORC,
     "    check_cross_fib_limits(sub_pool, sorted_events, tables)",
     "    check_cross_fib_limits(sub_pool, sorted_events, None)"),
    ("N25_sweep_log_drops_the_moves", SW,
     "                + (f\" moves={len(limit.price_moves)}\" if limit.price_moves else \"\")   # PLAN_M: a moving P",
     "                + \"\""),
]
