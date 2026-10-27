"""PLAN_T T1 mutants — run from the repo root:
    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \
        engine_v2/plans/qa_cycle0_carveout_inputs/mutants_plan_t_t1.py --out %TEMP%/mtT1

Each row: (id, relpath, old, new) — `old` occurs exactly once (text mode, LF anchors). The sketch after each = the
pin expected to kill it (written BEFORE the run).
"""
UP = "engine_v2/structure/unified_probe.py"
EDM = "engine_v2/multitf/entity_df_mutation.py"
CFL = "engine_v2/multitf/cross_fib_limit.py"
SW = "engine_v2/multitf/lifecycle_sweep.py"
ORC = "engine_v2/pipeline/orchestrator.py"

MUT = [
    # the probe's MS: the caller's timeframe → the helper's H1 default: test_phase2_ms_gets_the_callers_timeframe_values[M15/M5]
    ("t1_tf_dropped", UP, "            timeframe=timeframe,\n            pip_size=pip_size,\n",
     "            pip_size=pip_size,\n"),
    # a hard-coded M15: the [M5] and [H1] parametrizations
    ("t1_tf_hardcoded", UP, "            timeframe=timeframe,\n            pip_size=pip_size,\n",
     "            timeframe=\"M15\",\n            pip_size=pip_size,\n"),
    # pip size → MS's default 0.0001: test_phase2_ms_gets_the_pairs_pip_size[USD_JPY]
    ("t1_pip_dropped", UP, "            pip_size=pip_size,\n            cross_fib_limit=cross_fib_limit,\n",
     "            cross_fib_limit=cross_fib_limit,\n"),
    # a fresh "no P" instead of the handed limit: the identity pins (handed object; PLAN_I F7's rewritten pin)
    ("t1_limit_fresh", UP, "            pip_size=pip_size,\n            cross_fib_limit=cross_fib_limit,\n",
     "            pip_size=pip_size,\n            cross_fib_limit=CrossFibLimit(None, \"none\"),\n"),
    # the REQUIRED-with-Phase-2 assert dropped: None reaches the helper's isinstance assert, whose message lacks
    # "cross-fib limit" → test_phase2_without_a_limit_raises_and_phase1_needs_none (match=) fails
    ("t1_assert_dropped", UP, "    assert cross_fib_limit is not None, (\n"
     "        \"[unified_probe] Phase 2's MS needs the sub's cross-fib limit (PLAN_T T1)\")\n", ""),
    # the cache hands None on a miss: test_the_probe_cache_calls_the_limit_on_a_miss_only (seen == [None])
    ("t1_cache_none", EDM, "        cross_fib_limit=(cross_fib_limit() if cross_fib_limit is not None else None),\n",
     "        cross_fib_limit=None,\n"),
    # the cache hands the callable itself, uncalled: the same pin (calls == [], seen == [fn])
    ("t1_cache_uncalled", EDM, "        cross_fib_limit=(cross_fib_limit() if cross_fib_limit is not None else None),\n",
     "        cross_fib_limit=cross_fib_limit,\n"),
    # the FC resolver drops the limit: test_the_dispatcher_and_the_fc_resolver_pass_the_limit_to_the_probe_cache
    ("t1_resolver_drops", EDM, "        cross_fib_limit=cross_fib_limit,   # the sub's limit for Phase 2's MS (PLAN_T T1), lazy\n",
     "        cross_fib_limit=None,\n"),
    # the dispatcher drops it: the same pin
    ("t1_dispatcher_drops", EDM,
     "            trigger, parent_df, m15_df, pool=pool, parent_path=parent_path, cross_fib_limit=cross_fib_limit,\n",
     "            trigger, parent_df, m15_df, pool=pool, parent_path=parent_path,\n"),
    # the orchestrator hands no limit: test_the_orchestrator_hands_a_first_confluence_its_limit_and_the_sweep_its_check
    ("t1_orch_no_limit", ORC, "                    if t.trigger_type == \"first_confluence\" else None)\n",
     "                    if False else None)\n"),
    # the orchestrator hands the sweep no check: the same pin (the sweep stub REQUIRES the keyword)
    ("t1_orch_no_check", ORC, "        check_probe_sub_consistency=check_probe_sub_consistency,\n    )\n", "    )\n"),
    # the limit helper's direction assert dropped: test_a_counter_direction_first_confluence_asserts
    ("t1_dir_assert_dropped", CFL, "    assert int(trigger.direction) == int(parent_tables.parent_sd[S]), (\n",
     "    assert True, (\n"),
    # the limit helper reads another trigger type: the C>=1 pin (the x−1 branch → NO_LIMIT, not parent_cts_threshold)
    ("t1_helper_wrong_type", CFL, "            \"first_confluence\", trigger.cycle, int(trigger.direction), int(trigger.trigger_idx),\n",
     "            \"subsequent_confluence\", trigger.cycle, int(trigger.direction), int(trigger.trigger_idx),\n"),
    # --- the tripwire (rewritten after the landing review: two branches) ---
    # inverted: test_the_consistency_tripwire (the equal rows warn)
    ("t1_trip_inverted", EDM, "        disagree = own != fin\n", "        disagree = own == fin\n"),
    # cache hits checked too: the cache_hit row warns
    ("t1_trip_cache_hit", EDM, "    if trigger.trigger_type != \"first_confluence\" or res.cache_hit:\n",
     "    if trigger.trigger_type != \"first_confluence\":\n"),
    # every other condition flagged: the end_idx_reached row warns
    ("t1_trip_any_cond", EDM, "    else:\n        return\n    if disagree:\n",
     "    else:\n        own, what, disagree = None, \"?\", True\n    if disagree:\n"),
    # slice-local 2nd-CTS moment: the equal row warns
    ("t1_trip_local", EDM, "        own = None if len(est) < 2 else int(_second_cts_moment(est)) + int(slice_begin)\n",
     "        own = None if len(est) < 2 else int(_second_cts_moment(est))\n"),
    # list order, not anchor order: the reversed-order row warns
    ("t1_trip_list_order", EDM, "        est = _collect_cts_established(bounded.events, structure_id=0)\n",
     "        est = [e for e in bounded.events if e.type == \"CTS_ESTABLISHED\"]\n"),
    # slice-local cycle-0 confirmation: the "= the probe's" row (10 < 110) warns
    ("t1_trip_nr_local", EDM, "        own = None if conf is None else int(conf.idx) + int(slice_begin)\n",
     "        own = None if conf is None else int(conf.idx)\n"),
    # <= : the "= the probe's" row warns
    ("t1_trip_nr_le", EDM, "        disagree = own is not None and own < fin\n",
     "        disagree = own is not None and own <= fin\n"),
    # != : the "sub after" row warns
    ("t1_trip_nr_ne", EDM, "        disagree = own is not None and own < fin\n",
     "        disagree = own is not None and own != fin\n"),
    # the no_retrace branch never fires: the "sub confirmed at 110" row stays silent
    ("t1_trip_nr_off", EDM, "    elif cond == \"no_retrace\":\n", "    elif cond == \"no_retrace_never\":\n"),
    # --- the landing review's misses ---
    # the factory evaluates the formula eagerly: test_the_first_confluence_limit_is_lazy (building raises)
    ("t1_factory_eager", CFL, "    def limit() -> CrossFibLimit:\n        return cross_fib_limit_for(\n",
     "    _now = cross_fib_limit_for(\n"
     "        \"first_confluence\", trigger.cycle, int(trigger.direction), int(trigger.trigger_idx),\n"
     "        h1_events=h1_events, parent_tables=parent_tables, reversing_sub=None, prev_sub=None)\n\n"
     "    def limit() -> CrossFibLimit:\n        return cross_fib_limit_for(\n"),
    # the orchestrator hands the factory no events: the wiring pin's parent-cycle-1 limit
    ("t1_orch_no_events", ORC,
     "        fc_limit = (first_confluence_probe_limit(t, h1_events=sorted_events, parent_tables=tables)\n",
     "        fc_limit = (first_confluence_probe_limit(t, h1_events=[], parent_tables=tables)\n"),
    # the sweep does not hand its log: the sweep test's check requires `log`
    ("t1_sweep_no_log", SW, "            self.check_probe_sub_consistency(t, res, sub, log=self.log)\n",
     "            self.check_probe_sub_consistency(t, res, sub)\n"),
    # the sweep checks on a pool hit too: test_the_sweep_calls_the_check_once_per_sub_created
    ("t1_sweep_on_hit", SW, "        if created and self.check_probe_sub_consistency is not None:\n",
     "        if self.check_probe_sub_consistency is not None:\n"),
]
