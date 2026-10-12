"""PLAN_M landing 2 — the landing review's surviving mutants (lens B: X01 the closure's parent cycle, X04 / X07
the two R6-2 (a) shapes, X05 the collapse key) + the fix-up's own (M2R1 the tripwire's counter-first lookup,
M2R2 the trigger-type validation), re-run after the fix-up pins (PLAN_M §15). All must be KILLED."""

MUT = [
    ('X01_closure_cycle_zero', 'engine_v2/pipeline/orchestrator.py',
     '            t.trigger_type, t.cycle, int(t.direction), int(record_start),',
     '            t.trigger_type, (t.parent_sid, 0), int(t.direction), int(record_start),'),
    ('X04_tripwire_creating_parent', 'engine_v2/multitf/cross_fib_limit.py',
     'first.trigger_type, (first.parent_sid, first.parent_cycle_id), int(sub.direction), int(sub.start_idx),',
     'first.trigger_type, (creating.parent_sid, creating.parent_cycle_id), int(sub.direction), int(sub.start_idx),'),
    ('X05_collapse_by_h1', 'engine_v2/multitf/cross_fib_limit.py',
     '    for _h1, m15, price, _et in steps:\n        if collapsed and collapsed[-1][0] == int(m15):',
     '    _ph = None\n    for _h1, m15, price, _et in steps:\n        _same, _ph = (_ph == _h1), _h1\n        if collapsed and _same:'),
    ('X07_tripwire_at_creating_start', 'engine_v2/multitf/cross_fib_limit.py',
     'first.trigger_type, (first.parent_sid, first.parent_cycle_id), int(sub.direction), int(sub.start_idx),',
     'first.trigger_type, (first.parent_sid, first.parent_cycle_id), int(sub.direction), int(creating.start_idx),'),
    ('M2R1_tripwire_looks_up_a_reversing_sub_for_a_counter_sub', 'engine_v2/multitf/cross_fib_limit.py',
     '        if first.trigger_type == "reversal" and not counter:',
     '        if first.trigger_type == "reversal":'),
    ('M2R2_no_trigger_type_validation', 'engine_v2/multitf/cross_fib_limit.py',
     '    assert trigger_type in _TRIGGER_TYPES, f"[cross_fib_limit] unknown trigger type {trigger_type!r}"',
     '    pass'),
]
