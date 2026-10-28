"""PLAN_X X1 mutants: `REVERSAL_CANDIDATE.meta["pattern"]` + the `[RV_SCHEDULE]` / `[RV_APPLY]` lines name the pattern.

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py engine_v2/plans/plan_x_inputs/mutants_x1.py

Killers (`tests/test_plan_x_reversal_candidate_pattern.py`): `test_the_candidate_names_the_pattern_its_reversal_applies`
(both paths), `test_the_rv_schedule_and_rv_apply_lines_name_the_pattern`. LF anchors.
"""
MS = "engine_v2/structure/market_structure.py"
MUT = [
    # the old getattr chain back at the schedule site -> every candidate says "?" (the harm the pin shows)
    ("x1_sched_old", MS, "            pat = ev_r.name\n",
     '            pat = getattr(ev_r, "pat", None) or getattr(ev_r, "pattern", None) or "?"\n'),
    # the old chain back at the apply site -> the [RV_APPLY] line says pat=?
    ("x1_apply_old", MS, "        pat = st.pending_reversal_ev.name\n",
     '        pat = getattr(st.pending_reversal_ev, "pat", None) or getattr(st.pending_reversal_ev, "pattern", None) or "?"\n'),
    # a constant name -> the candidate disagrees with the reversal it schedules on at least one path
    ("x1_sched_const", MS, "            pat = ev_r.name\n", '            pat = "continuous"\n'),
    # the meta key fed from somewhere else (the scheduled direction) -> not the pattern's name
    ("x1_meta_wrong", MS, '                        "pattern": pat,\n',
     '                        "pattern": str(int(self.struct_direction)),\n'),
]
