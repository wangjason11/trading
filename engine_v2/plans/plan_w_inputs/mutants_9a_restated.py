"""PLAN_W 9a — PLAN_S step 2's mutants on LIVE code that 9a moved (`anchor_scan_9a.txt`), re-anchored on the new code:
the positional scan of `is_filled` lost one indent level (its `if contiguous:` block is gone) and the cache's
`_idx_from_pos(confirmed_pos)` became `confirmed_pos`. Derived from `qa_ic_revisit_inputs/mutants_plan_s_step2.py`, so
each mutation is PLAN_S's own. (Its six anchors on the deleted label-based fallback have no code left to mutate.)

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py engine_v2/plans/plan_w_inputs/mutants_9a_restated.py
"""
import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "plan_s_step2", os.path.join(_HERE, "..", "qa_ic_revisit_inputs", "mutants_plan_s_step2.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
_SRC = {mid: (rel, old, new) for mid, rel, old, new in _m.MUT}


def _dedent4(s):
    return "\n".join(line[4:] if line.startswith("    ") else line for line in s.split("\n"))


MUT = []
for mid in ("s2_fast_bull", "s2_fast_bear", "s2_latch_fast_bull", "s2_latch_fast_bear"):
    rel, old, new = _SRC[mid]
    MUT.append((mid + "_9a", rel, _dedent4(old), _dedent4(new)))
rel, old, new = _SRC["s2_cache_bull_offset_only"]
MUT.append(("s2_cache_bull_offset_only_9a", rel,
            old.replace("_idx_from_pos(confirmed_pos)", "confirmed_pos"),
            new.replace("_idx_from_pos(confirmed_pos)", "confirmed_pos")))
