"""PLAN_V 8b mutants (§2.3): the phase-3 pick and the phase-1 re-point of `multitf/lifecycle_sweep.py`.

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py engine_v2/plans/plan_v_inputs/mutants_8b.py

Killers (`tests/test_lifecycle_sweep_unit.py`): `test_a_same_candle_replacement_chain_names_the_replacer_that_survives`,
`test_three_co_starters_all_name_the_last_one`, `test_the_re_point_leaves_a_record_ended_at_an_earlier_candle`.
Anchors use LF (the runner reads text mode).
"""
SW = "engine_v2/multitf/lifecycle_sweep.py"
_IF = "                    if r.trigger_end_idx == int(t) and r.ended_by_pool_idx == inc.pool_idx:"

MUT = [
    # phase 3 back to the FIRST of the highest-priority entries (today's behaviour)
    ("v8b_phase3_first_entry", SW,
     "best = [e for e in entries if _END_REASON_PRIORITY.get(e[1], 99) == top][-1]",
     "best = [e for e in entries if _END_REASON_PRIORITY.get(e[1], 99) == top][0]"),
    # the re-point dropped
    ("v8b_no_repoint", SW,
     "                        r.ended_by_pool_idx = rec.pool_idx\n",
     "                        pass\n"),
    # the re-point without its "at this candle" filter
    ("v8b_repoint_any_candle", SW, _IF,
     "                    if r.ended_by_pool_idx == inc.pool_idx:"),
    # the re-point without its "frozen by inc" filter
    ("v8b_repoint_any_ender", SW, _IF,
     "                    if r.trigger_end_idx == int(t):"),
]
