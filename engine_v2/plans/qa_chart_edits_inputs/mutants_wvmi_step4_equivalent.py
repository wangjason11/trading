"""The EQUIVALENT mutants of "WVMI back on the chart" step 4 (2026-10-26) — expected to SURVIVE the full suite; run
anyway (memory `feedback_cold_review_plans.md`, the PLAN_Q rule: run every "equivalent" mutant and fix the claim when
one is not). The killable ones: `mutants_wvmi_step4.py`. Run from the repo root:

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \
        engine_v2/plans/qa_chart_edits_inputs/mutants_wvmi_step4_equivalent.py

(the runner exits 1 on a survivor — here that is the expected outcome).

- `temp_status_idx_next_lp`: a temporary row's status candle = the next LP's `lp_idx` instead of its `as_of_idx`. Only
  the FIRST LP can be as of a later candle (the CTS_CONFIRMED moment), and the first LP is never a "next" one — every
  later LP is as of itself (`_temporary_lp_sequence` appends `(i, i)`).
- `h1_relative_dir_empty_string`: an H1 row's `relative_dir` "" instead of None — both write an empty CSV cell.
"""
WVMI = "engine_v2/zones/wvmi.py"
EXP = "engine_v2/debug/export_wvmi.py"

MUT = [
    ("temp_status_idx_next_lp", WVMI,
     "        rows = [row(as_of, lp, \"temporary\", \"replaced by a further close\", lp_seq[k + 1][0])\n",
     "        rows = [row(as_of, lp, \"temporary\", \"replaced by a further close\", lp_seq[k + 1][1])\n"),
    ("h1_relative_dir_empty_string", EXP,
     "                    if sid_rec is not None else None\n",
     "                    if sid_rec is not None else \"\"\n"),
]
