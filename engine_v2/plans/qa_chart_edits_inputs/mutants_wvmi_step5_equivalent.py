"""The EQUIVALENT mutants of "WVMI back on the chart" step 5 (2026-10-26) — run them (the PLAN_Q rule); each is
expected to SURVIVE, and why:

- `m15_none_idx_lookup`: `time_by_idx.get(None)` is None anyway (the dict's keys are ints), so dropping the
  `m15_idx is not None` guard changes nothing. Run 2026-10-26: SURVIVED (2035 passed, 3 skipped).

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \\
        engine_v2/plans/qa_chart_edits_inputs/mutants_wvmi_step5_equivalent.py
"""
EM = "engine_v2/charting/export_m15_chart.py"

MUT = [
    ("m15_none_idx_lookup", EM, "        t = time_by_idx.get(m15_idx) if m15_idx is not None else None\n",
     "        t = time_by_idx.get(m15_idx)\n"),
]
