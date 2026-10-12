"""PLAN_M landing 1 — the landing review's (lens B) surviving mutants, re-run after the fix-up pins (PLAN_M §13):
X01 / X02 (a TRUE fix re-read live; `test_a_cycle_established_as_a_cross_keeps_it` with a narrowing move) and X04
(the projection call site skips localizing a no-break limit; `test_the_projection_hands_the_drawn_tracker_the_
slice_local_moves`); + M25, the review fix-up's new raise (A NOTE-8: a refusal at the establishment over an
active pre-established cross). All must be KILLED. Same runner as `mutants_m1.py`."""
FT = "engine_v2/zones/fib_tracker.py"
EDM = "engine_v2/multitf/entity_df_mutation.py"

MUT = [
    ('X01_true_fix_reread_live', FT,
     '        fixed = self._cross_permission.get((int(sid), int(target_cycle)))\n        if fixed is not None:\n            return fixed\n        if lim.price_moves',
     '        fixed = self._cross_permission.get((int(sid), int(target_cycle)))\n        if fixed is False or (fixed is True and self._evaluated_at is None):\n            return fixed\n        if lim.price_moves'),
    ('X02_ref_true_fix_reread_live', FT,
     '        reaches = fixed if fixed is not None else self._limit_reaches(int(sid), int(target_cycle), at)',
     '        reaches = fixed if fixed is False else self._limit_reaches(int(sid), int(target_cycle), at)'),
    ('X04_projection_skips_moves', EDM,
     '        cross_fib_limits={0: lim.localized(slice_begin)},',
     '        cross_fib_limits={0: lim.localized(slice_begin) if lim.break_pattern_anchor_idx is not None else lim},'),
    ("M25_refusal_over_cross_unchecked", FT,
     "        if not allowed and latest is not None and latest[1].active:",
     "        if False:"),
]
