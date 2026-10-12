"""PLAN_N landing review — the reviewer's three surviving mutants (LR1–LR3) + the hover pass-3 fix, re-run after the
fold (full suite each; every one must be KILLED).

SUPERSEDED on the code since `9f9406d` (the hover follow-up, option B, 2026-10-12): LR1's and H1's anchors are gone
(re-running this file reports them BAD). Use `mutants_hover_followup.py` — LR1 re-anchored, LR2 / LR3, and B's own
mutants; H1 (restore the all-structures snapshot) is EQUIVALENT under B. This file stays as the record of PLAN_N's run.

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \
        engine_v2/plans/plan_n_inputs/mutants_n_review.py --jobs 4 --out <a short %TEMP% dir>
"""
EP = "engine_v2/charting/export_plotly.py"

MUT = [
    ("LR1_hover_order_back_on_an_idx_sort", EP,
     "        all_range_events = _range_events_chart_order(structure_events)   # the box builder's order",
     "        all_range_events = sorted(range_started + range_updated + range_reset,\n"
     "                                  key=lambda e: (int(e.idx), _RANGE_TYPE_PRIORITY[e.type]))"),
    ("LR2_guard_not_per_structure", EP,
     '        sid = int(ev.meta.get("structure_id", 0))\n        idx = int(ev.idx)\n        if t == "RANGE_STARTED":',
     '        sid = 0\n        idx = int(ev.idx)\n        if t == "RANGE_STARTED":'),
    ("LR3_guard_takes_a_later_reset", EP,
     "            if r is not None and r <= idx:\n",
     "            if r is not None:\n"),
    # H1 belongs to the hover follow-up (`followup_hover_per_structure.md`): it ran against the per-structure fix
    # (KILLED, see the result file) and moves with it; against this landing's original pass 3 its anchor is absent.
    ("H1_hover_snapshot_of_every_structure_restored", EP,
     "            snaps_by_sid.setdefault(sid, {})[idx] = dict(rng_now) if rng_now is not None else None\n",
     "            for _s in set(snaps_by_sid) | set(open_ranges) | {sid}:\n"
     "                snaps_by_sid.setdefault(_s, {})[idx] = dict(open_ranges[_s]) if _s in open_ranges else None\n"),
]
