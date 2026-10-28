"""PLAN_W 9b mutants (§2.4): `zones/poi_zones._poi_record_gate`.

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py engine_v2/plans/plan_w_inputs/mutants_9b.py

Killers (`tests/test_institutional_pattern.py`): `test_records_gate_none_rows_and_tiebreak_on_the_pick`,
`test_a_cycle_that_ends_before_its_fib_locks_gets_a_poi_ending_at_the_cycle_end`. Anchors use LF.
"""
PZ = "engine_v2/zones/poi_zones.py"
_GATE = "    return bool(fib_state.active or fib_state.locked)"

MUT = [
    # the old gate back: an ended, never-locked cycle gets no record
    ("v9b_old_gate", PZ, _GATE, "    return bool((fib_state.active and fib_state.end_idx is None) or fib_state.locked)"),
    # active AND locked: the live unlocked records lost
    ("v9b_and", PZ, _GATE, "    return bool(fib_state.active and fib_state.locked)"),
    # locked only
    ("v9b_locked_only", PZ, _GATE, "    return bool(fib_state.locked)"),
    # every record (the condition-inactive / superseded versions too)
    ("v9b_every_record", PZ, _GATE, "    return True"),
]
