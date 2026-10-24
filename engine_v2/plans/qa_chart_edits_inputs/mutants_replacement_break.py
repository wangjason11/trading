"""Mutants for the pool-wide replacer of a replaced sub's chart break (SESSION ORDER item 5, 2026-10-24 —
`user_spec_20261024_replacement_break.md`). Run from the repo root:

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \
        engine_v2/plans/qa_chart_edits_inputs/mutants_replacement_break.py

`old` / `new` use LF (the runner reads in text mode). Every mutant must be KILLED by the full suite.
"""
SWEEP = "engine_v2/multitf/lifecycle_sweep.py"
POOL = "engine_v2/multitf/sub_structure_pool.py"
EDM = "engine_v2/multitf/entity_df_mutation.py"
ORCH = "engine_v2/pipeline/orchestrator.py"
SIDS = "engine_v2/multitf/sid_records.py"
CHART = "engine_v2/charting/export_m15_chart.py"

MUT = [
    # the sweep's phase 4 — the sub keeps its winning record's replacer
    ("phase4_not_set", SWEEP,
     "        sub.ended_by_pool_idx = r.ended_by_pool_idx\n", ""),
    ("phase4_wrong_record", SWEEP,
     "        sub.ended_by_pool_idx = r.ended_by_pool_idx\n",
     "        sub.ended_by_pool_idx = live[-1].ended_by_pool_idx\n"),
    # the numbering — pool_idx -> the final sub_id, on the sub as on its records
    ("number_self_id", POOL,
     "            s.ended_by_sub_id = (final_by_pool_idx.get(s.ended_by_pool_idx)\n",
     "            s.ended_by_sub_id = (final_by_pool_idx.get(s.pool_idx)\n"),
    ("number_dropped", POOL,
     "            s.ended_by_sub_id = (final_by_pool_idx.get(s.ended_by_pool_idx)\n"
     "                                 if s.ended_by_pool_idx is not None else None)\n", ""),
    # the driver — the replacer's anchor, looked up in the pool
    ("driver_own_anchor", ORCH,
     "            ended_by_starting_idx=(int(replacer.starting_idx) if replacer is not None else None),\n",
     "            ended_by_starting_idx=(int(sub.starting_idx) if replacer is not None else None),\n"),
    # the projection — the pairing asserts and the meta
    ("proj_assert_reason_dropped", EDM,
     '    assert (sub.end_reason == "same_dir_replacement") == (sub.ended_by_sub_id is not None), (\n',
     "    assert True, (\n"),
    ("proj_assert_anchor_dropped", EDM,
     "    assert (sub.ended_by_sub_id is None) == (ended_by_starting_idx is None), (\n",
     "    assert True, (\n"),
    ("proj_meta_anchor_none", EDM,
     '            "ended_by_starting_idx": ended_by_starting_idx,\n',
     '            "ended_by_starting_idx": None,\n'),
    ("proj_meta_id_none", EDM,
     '            "ended_by_sub_id": sub.ended_by_sub_id,\n',
     '            "ended_by_sub_id": None,\n'),
    # the SidRecord — the passthrough
    ("sid_meta_anchor_dropped", SIDS,
     '                "ended_by_starting_idx": m.get("ended_by_starting_idx"),\n',
     '                "ended_by_starting_idx": None,\n'),
    # the chart — reads the sub's replacer, loudly, and draws it
    ("chart_own_anchor", CHART,
     '    anchor = sid_rec.meta["ended_by_starting_idx"]\n',
     "    anchor = sid_rec.creation_event_idx\n"),
    ("chart_silent_get", CHART,
     '    anchor = sid_rec.meta["ended_by_starting_idx"]\n',
     '    anchor = sid_rec.meta.get("ended_by_starting_idx")\n    if anchor is None:\n        return None\n'),
    ("chart_reason_guard_dropped", CHART,
     '    if (getattr(sid_rec, "end_reason", None) or "") != "same_dir_replacement":\n        return None\n'
     '    anchor = sid_rec.meta["ended_by_starting_idx"]\n',
     '    anchor = sid_rec.meta["ended_by_starting_idx"]\n'),
    ("chart_break_not_drawn", CHART,
     "    if _last_seq:\n        _bp = _replacement_break_point(",
     "    if False:\n        _bp = _replacement_break_point("),
]
