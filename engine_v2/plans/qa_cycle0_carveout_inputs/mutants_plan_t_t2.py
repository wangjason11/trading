"""PLAN_T T2 mutants — run from the repo root:
    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \
        engine_v2/plans/qa_cycle0_carveout_inputs/mutants_plan_t_t2.py --out %TEMP%/mtT2

Each row: (id, relpath, old, new) — `old` occurs exactly once (text mode, LF anchors). The sketch after each = the
pin expected to kill it (written BEFORE the run; the pins in tests/test_plan_t_t2_cycle0_carveout.py).
"""
MS = "engine_v2/structure/market_structure.py"
SE = "engine_v2/structure/structure_engine.py"
UP = "engine_v2/structure/unified_probe.py"
EDM = "engine_v2/multitf/entity_df_mutation.py"
SW = "engine_v2/multitf/lifecycle_sweep.py"

GATE = "            and (st.cts_cycle_id > 0 or not self.cycle0_proximity_blocked)\n"

MUT = [
    # --- the gate ---
    # the carve-out gone everywhere: test_the_gate_reads_the_flag_on_cycle_0_only[True-0-False]
    ("t2_gate_no_carveout", MS, GATE, "            and True\n"),
    # the flag inverted: the (True,0) and (False,0) rows
    ("t2_gate_no_not", MS, GATE, "            and (st.cts_cycle_id > 0 or self.cycle0_proximity_blocked)\n"),
    # the flag read on every cycle: the (True,1) row must fire
    ("t2_gate_every_cycle", MS, GATE, "            and (not self.cycle0_proximity_blocked)\n"),
    # the universal carve-out back: the (False,0) row; the real cycle 0 unblocked
    ("t2_gate_old_rule", MS, GATE, "            and st.cts_cycle_id > 0\n"),
    # the narrow-gap boundary `>=` → `>`: test_the_narrow_gap_boundary_is_inclusive[0.0-True] (an exact gap)
    ("t2_gap_gt", MS, "            >= self._min_gap_threshold\n", "            > self._min_gap_threshold\n"),
    # MS's default flipped: test_direct_construction_keeps_the_pre_plan_t_rule
    ("t2_ms_default", MS, "        cycle0_proximity_blocked: bool = True,\n    ):\n",
     "        cycle0_proximity_blocked: bool = False,\n    ):\n"),
    # --- structure_engine ---
    # the builder ignores its flag: test_a_real_cycle_0_confirms_by_proximity_only_when_unblocked (unblocked → blocked)
    ("t2_builder_const", SE, "        cycle0_proximity_blocked=bool(cycle0_proximity_blocked),\n        **kwargs,\n",
     "        cycle0_proximity_blocked=True,\n        **kwargs,\n"),
    # the main blocks every sid: test_compute_structure_blocks_sid_0_and_not_the_reversal (sid 1 False)
    ("t2_main_all", SE, "            cycle0_proximity_blocked=(structure_id == 0),\n",
     "            cycle0_proximity_blocked=True,\n"),
    # the main blocks no sid: the same pin (sid 0 True)
    ("t2_main_none", SE, "            cycle0_proximity_blocked=(structure_id == 0),\n",
     "            cycle0_proximity_blocked=False,\n"),
    # the bounded run defaults to blocked: test_compute_bounded_structure_defaults_to_unblocked
    ("t2_bounded_default", SE, "    cycle0_proximity_blocked: bool = False,\n) -> BoundedStructureResult:\n",
     "    cycle0_proximity_blocked: bool = True,\n) -> BoundedStructureResult:\n"),
    # the bounded run ignores its flag: the real cycle 0 blocked must not confirm by proximity
    ("t2_bounded_const", SE, "        cycle0_proximity_blocked=cycle0_proximity_blocked,\n    )\n    ms.debug = True\n",
     "        cycle0_proximity_blocked=False,\n    )\n    ms.debug = True\n"),
    # --- the probe ---
    # unified_probe → _run_phase2 drops it: test_phase2_ms_gets_the_flag[True]
    ("t2_probe_to_phase2", UP, "        cross_fib_limit=cross_fib_limit,\n        cycle0_proximity_blocked=cycle0_proximity_blocked,\n    )\n",
     "        cross_fib_limit=cross_fib_limit,\n        cycle0_proximity_blocked=False,\n    )\n"),
    # _run_phase2 → its MS drops it: the same pin
    ("t2_phase2_to_ms", UP, "            cross_fib_limit=cross_fib_limit,\n            cycle0_proximity_blocked=cycle0_proximity_blocked,\n        )\n",
     "            cross_fib_limit=cross_fib_limit,\n            cycle0_proximity_blocked=False,\n        )\n"),
    # --- entity_df_mutation ---
    # the probe cache drops it: test_the_probe_cache_and_the_fc_resolver_pass_the_flag
    ("t2_cache_drops", EDM, "        cross_fib_limit=(cross_fib_limit() if cross_fib_limit is not None else None),\n"
     "        cycle0_proximity_blocked=cycle0_proximity_blocked,\n",
     "        cross_fib_limit=(cross_fib_limit() if cross_fib_limit is not None else None),\n"
     "        cycle0_proximity_blocked=False,\n"),
    # the FC resolver drops it: the same pin
    ("t2_resolver_drops", EDM, "        cycle0_proximity_blocked=cycle0_proximity_blocked,   # = its sub's carve-out flag (PLAN_T T2)\n",
     "        cycle0_proximity_blocked=False,\n"),
    # the dispatcher drops it: the same pin
    ("t2_dispatch_drops", EDM, "            cycle0_proximity_blocked=cycle0_proximity_blocked,\n        )\n    assert not cycle0_proximity_blocked",
     "            cycle0_proximity_blocked=False,\n        )\n    assert not cycle0_proximity_blocked"),
    # the dispatcher's sibling assert dropped: test_a_sibling_trigger_with_the_flag_asserts
    ("t2_dispatch_assert", EDM,
     "    assert not cycle0_proximity_blocked, f\"[entity_compute] {uc}: only a first_confluence keeps the carve-out\"\n", ""),
    # the geometry core drops it: test_the_geometry_core_hands_the_flag_to_the_bounded_run
    ("t2_core_drops", EDM, "            cycle0_proximity_blocked=cycle0_proximity_blocked,   # PLAN_T T2 — the creating trigger's\n",
     "            cycle0_proximity_blocked=False,\n"),
    # the pool builder hands the core a constant: test_the_pool_builder_stores_and_guards_the_flag (handed == [True])
    ("t2_bog_const", EDM, "        cycle0_proximity_blocked=bool(cycle0_proximity_blocked),\n    )\n    if geom is None:\n",
     "        cycle0_proximity_blocked=False,\n    )\n    if geom is None:\n"),
    # the pool builder does not store it: the same pin (sub.cycle0_proximity_blocked)
    ("t2_bog_no_store", EDM, "    sub.cycle0_proximity_blocked = bool(cycle0_proximity_blocked)\n", ""),
    # a pool hit accepts a flag: the same pin (the hit assert)
    ("t2_bog_hit_open", EDM, "        assert cycle0_proximity_blocked is None, (\n", "        assert True, (\n"),
    # --- the predicate ---
    ("t2_pred_no_type", SW, "    return (trigger_type == \"first_confluence\" and tuple(parent_cycle) == (0, 0)\n",
     "    return (tuple(parent_cycle) == (0, 0)\n"),
    ("t2_pred_cycle_01", SW, "    return (trigger_type == \"first_confluence\" and tuple(parent_cycle) == (0, 0)\n",
     "    return (trigger_type == \"first_confluence\" and tuple(parent_cycle) == (0, 1)\n"),
    ("t2_pred_no_cycle", SW, "    return (trigger_type == \"first_confluence\" and tuple(parent_cycle) == (0, 0)\n",
     "    return (trigger_type == \"first_confluence\"\n"),
    ("t2_pred_no_path", SW, "            and parent_path == \"H1.main\")\n", "            )\n"),
    # --- the sweep ---
    # the probe gets a constant: test_the_sweep_hands_one_evaluation_to_the_probe_and_the_geometry
    ("t2_sweep_probe_const", SW, "            res = self.resolve_start(t, int(t.trigger_idx), cycle0_proximity_blocked=blocked)\n",
     "            res = self.resolve_start(t, int(t.trigger_idx), cycle0_proximity_blocked=False)\n"),
    # a pool hit gets the flag too: the same pin ((454, None))
    ("t2_sweep_geom_on_hit", SW, "                                    cycle0_proximity_blocked=(blocked if miss else None))\n",
     "                                    cycle0_proximity_blocked=blocked)\n"),
    # the creator tripwire off: test_the_creator_tripwire
    ("t2_trip_off", SW, "        if blocked and (res.cache_hit or not created):\n", "        if False:\n"),
    # the creator tripwire ignores cache hits: test_the_creator_tripwire[True-False]
    ("t2_trip_no_cache", SW, "        if blocked and (res.cache_hit or not created):\n", "        if blocked and (not created):\n"),
    # --- the landing review's misses ---
    # the orchestrator's resolver hands a constant: test_the_orchestrator_hands_..._its_check (flags == [True, False])
    ("t2_orch_resolve_const", "engine_v2/pipeline/orchestrator.py",
     "            cross_fib_limit=fc_limit, cycle0_proximity_blocked=cycle0_proximity_blocked,\n",
     "            cross_fib_limit=fc_limit, cycle0_proximity_blocked=False,\n"),
    # the orchestrator's geometry builder hands a constant: the same pin (built == [True, False, None])
    ("t2_orch_geom_const", "engine_v2/pipeline/orchestrator.py",
     "            cross_fib_limit=cross_fib_limit, cycle0_proximity_blocked=cycle0_proximity_blocked,\n",
     "            cross_fib_limit=cross_fib_limit, cycle0_proximity_blocked=True,\n"),
    # hard-coded True on each hop: the [False] parametrizations
    ("t2_cache_true", EDM, "        cross_fib_limit=(cross_fib_limit() if cross_fib_limit is not None else None),\n"
     "        cycle0_proximity_blocked=cycle0_proximity_blocked,\n",
     "        cross_fib_limit=(cross_fib_limit() if cross_fib_limit is not None else None),\n"
     "        cycle0_proximity_blocked=True,\n"),
    ("t2_resolver_true", EDM, "        cycle0_proximity_blocked=cycle0_proximity_blocked,   # = its sub's carve-out flag (PLAN_T T2)\n",
     "        cycle0_proximity_blocked=True,\n"),
    ("t2_dispatch_true", EDM, "            cycle0_proximity_blocked=cycle0_proximity_blocked,\n        )\n    assert not cycle0_proximity_blocked",
     "            cycle0_proximity_blocked=True,\n        )\n    assert not cycle0_proximity_blocked"),
    ("t2_bog_true", EDM, "        cycle0_proximity_blocked=bool(cycle0_proximity_blocked),\n    )\n    if geom is None:\n",
     "        cycle0_proximity_blocked=True,\n    )\n    if geom is None:\n"),
    ("t2_bog_store_true", EDM, "    sub.cycle0_proximity_blocked = bool(cycle0_proximity_blocked)\n",
     "    sub.cycle0_proximity_blocked = True\n"),
]
