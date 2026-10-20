"""The build session's run of the §1 reference simulator (PLAN_R §8): step 1 (parent zones only, every cycle) and the
overlay chains of (0,1) / (1,2) with the decided flags (pass_from="usable", src_qual=True). Log: `m4_run.log`.

Like the sims README says: copy `../sims/` into a scratchpad, run `review_measure/m1_frames.py` and `m2_h1.py` there
(from the repo root, PYTHONPATH = the repo root — they write the pickles `m4_sim.py` reads), then put this file into
that `review_measure/` copy and run it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m4_sim as m  # noqa: E402

print("=== step 1 (parent only) ===")
for k in sorted(m.CY):
    print(k, m.CY[k]["lo"], m.CY[k]["hi"], m.run(k, overlay=False, verbose=True))
print("=== overlay pass_from=usable src_qual=True ===")
for k in ((0, 1), (1, 2)):
    print(k, m.run(k, overlay=True, pass_from="usable", src_qual=True, verbose=True))
