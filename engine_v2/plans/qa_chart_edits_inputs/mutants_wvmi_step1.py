"""Mutants for "WVMI back on the chart" step 1 (2026-10-26, `user_spec_20261024_wvmi_on_chart.md`): `first_counter`
reads the H1 cycles' first sd zone-proximity trigger directly (`zone_proximity.first_sd_proximity_triggers`, shared
with the main WVMI gate), not the gated WVMI records. Run from the repo root:

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \
        engine_v2/plans/qa_chart_edits_inputs/mutants_wvmi_step1.py

`old` / `new` use LF (the runner reads in text mode). Every mutant must be KILLED by the full suite.
"""
PROX = "engine_v2/zones/zone_proximity.py"
UC1 = "engine_v2/multitf/uc1_trigger.py"
ORCH = "engine_v2/pipeline/orchestrator.py"
CHART = "engine_v2/charting/_zone_render.py"

MUT = [
    # the ONE reading of "a cycle's first sd trigger"
    ("helper_last_trigger", PROX,
     'return {key: trigs[0] for key, trigs in zone_proximity_triggers.items() if trigs and trigs[0].direction == "sd"}',
     'return {key: trigs[-1] for key, trigs in zone_proximity_triggers.items() if trigs and trigs[0].direction == "sd"}'),
    ("helper_any_direction", PROX,
     'return {key: trigs[0] for key, trigs in zone_proximity_triggers.items() if trigs and trigs[0].direction == "sd"}',
     'return {key: trigs[0] for key, trigs in zone_proximity_triggers.items() if trigs}'),
    ("helper_opp_sd", PROX,
     'return {key: trigs[0] for key, trigs in zone_proximity_triggers.items() if trigs and trigs[0].direction == "sd"}',
     'return {key: trigs[0] for key, trigs in zone_proximity_triggers.items() if trigs and trigs[0].direction == "opp_sd"}'),
    # the detector
    ("uc1_any_first_trigger", UC1,
     "        trig = first_sd.get(key)\n",
     "        trig = (zone_proximity_triggers.get(key) or [None])[0]\n"),
    ("uc1_event_idx_is_cts_moment", UC1,
     '                "trigger_event_idx": trig.idx,\n',
     '                "trigger_event_idx": cts_ev.idx,\n'),
    ("uc1_lists_order", UC1,
     "    for key, cts_ev in cts_conf_by_key.items():\n"
     "        trig = first_sd.get(key)\n"
     "        if trig is None:\n"
     "            continue\n",
     "    for key, trig in first_sd.items():\n"
     "        cts_ev = cts_conf_by_key.get(key)\n"
     "        if cts_ev is None:\n"
     "            continue\n"),
    # the gate reads the same helper
    ("gate_own_reading", ORCH,
     "        for key, first_sd in first_sd_proximity_triggers(zone_proximity_triggers).items()\n",
     "        for key, first_sd in {k: v[0] for k, v in zone_proximity_triggers.items() if v}.items()\n"),
    # the call sites
    ("dual_call_empty_lists", ORCH,
     "        sorted_events, h1_df, main_zone_proximity_triggers or {},\n",
     "        sorted_events, h1_df, {},\n"),
    ("chart_reads_nothing", CHART,
     '    zpt = attrs.get("zone_proximity_triggers") or {}\n',
     "    zpt = {}\n"),
]
