"""Mutants for "WVMI back on the chart" step 2 (2026-10-26, `user_spec_20261024_wvmi_on_chart.md`): the H1 WVMI
without its gate — `_compute_wvmi_records` is ungated for the main too, and the main's trigger attribution is a
post-pass (`orchestrator._stamp_main_wvmi_attribution`: each record's cycle's first zone-proximity trigger when sd,
else the same keys with None values). Run from the repo root:

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \
        engine_v2/plans/qa_chart_edits_inputs/mutants_wvmi_step2.py

`old` / `new` use LF (the runner reads in text mode). Every mutant must be KILLED by the full suite.
"""
ORCH = "engine_v2/pipeline/orchestrator.py"

_CALL = (
    "        if wvmi == \"first_sd_prox\":\n"
    "            _stamp_main_wvmi_attribution(wvmi_records, zone_proximity_triggers, structure_path_id)\n"
)
_STAMP = (
    "        stamp = attribution.get((rec.bos_structure_id, rec.bos_cycle_id), unattributed)\n"
    "        rec.meta = {**stamp, **{k: v for k, v in rec.meta.items() if k not in stamp}}\n"
)

MUT = [
    # the call site
    ("stamp_never", ORCH, _CALL,
     "        if False:\n"
     "            _stamp_main_wvmi_attribution(wvmi_records, zone_proximity_triggers, structure_path_id)\n"),
    ("stamp_subs_too", ORCH, _CALL,
     "        if wvmi != \"off\":\n"
     "            _stamp_main_wvmi_attribution(wvmi_records, zone_proximity_triggers, structure_path_id)\n"),
    ("gate_restored", ORCH, _CALL,
     _CALL
     + "            _keep = _first_sd_prox_attribution(zone_proximity_triggers, structure_path_id)\n"
     + "            wvmi_records = [r for r in wvmi_records if (r.bos_structure_id, r.bos_cycle_id) in _keep]\n"),
    # the stamp
    ("unattributed_no_keys", ORCH,
     "        stamp = attribution.get((rec.bos_structure_id, rec.bos_cycle_id), unattributed)\n",
     "        stamp = attribution.get((rec.bos_structure_id, rec.bos_cycle_id), {})\n"),
    ("unattributed_no_path", ORCH,
     "        \"structure_path_id\": structure_path_id,\n        \"trigger_inner\": None,\n",
     "        \"structure_path_id\": None,\n        \"trigger_inner\": None,\n"),
    ("stamp_keys_last", ORCH, _STAMP,
     "        stamp = attribution.get((rec.bos_structure_id, rec.bos_cycle_id), unattributed)\n"
     "        rec.meta = {**{k: v for k, v in rec.meta.items() if k not in stamp}, **stamp}\n"),
    ("stamp_shared_dict", ORCH, _STAMP,
     "        stamp = attribution.get((rec.bos_structure_id, rec.bos_cycle_id), unattributed)\n"
     "        rec.meta = stamp if not rec.meta else {**stamp, **{k: v for k, v in rec.meta.items() if k not in stamp}}\n"),
    ("stamp_key_swapped", ORCH,
     "        stamp = attribution.get((rec.bos_structure_id, rec.bos_cycle_id), unattributed)\n",
     "        stamp = attribution.get((rec.bos_cycle_id, rec.bos_structure_id), unattributed)\n"),
    # the attribution reads the ONE first-sd helper
    ("attribution_own_reading", ORCH,
     "        for key, first_sd in first_sd_proximity_triggers(zone_proximity_triggers).items()\n",
     "        for key, first_sd in {k: v[0] for k, v in zone_proximity_triggers.items() if v}.items()\n"),
]
