"""`reversal_shadow` over `random_tail_search` streams (2026-09-29d): wraps MarketStructure with the shadow FIRST, then
runs `random_tail_search run` (its own taps stack on top). From the repo root, with PYTHONPATH=. :

  REVERSAL_SHADOW_OUT=OUT.json python engine_v2/plans/plan_e_inputs/review_scripts/tails_shadow.py OUT.jsonl SEED N [LO HI]

RTS_BASES / RTS_NOINV are honoured as in random_tail_search. The shadow JSON is written at exit (its `agg` counts).
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reversal_shadow  # noqa: E402,F401  (wraps MS first)

out, seed, n, *rest = sys.argv[1:]
sys.argv = ["random_tail_search.py", "run", out, seed, n, *rest]
runpy.run_path(os.path.join(HERE, "random_tail_search.py"), run_name="__main__")
