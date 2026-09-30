"""Run the REAL replay through a scratch `engine_v2` tree (2026-09-29c; F3b variants, the cap and fallback-POI checks).

  cd <repo root> && REVERSAL_SHADOW_OUT=out.json python engine_v2/plans/plan_e_inputs/review_scripts/replay_variant.py <tree>

cwd must be the repo root (OANDA data via oanda.cfg, artifacts/); `<tree>/engine_v2` is put first on sys.path (asserted),
with `reversal_shadow` imported so every MS run of the variant is shadowed. Overwrites artifacts/debug + charts — compare
them (e.g. `cmp_save.py`) and re-run a normal replay before any save or /compare."""
import os, sys, runpy
tree = sys.argv[1]
sys.path.insert(0, tree)
sys.path.insert(1, os.path.join(os.getcwd(), "engine_v2", "plans", "plan_e_inputs", "review_scripts"))
import engine_v2.structure.market_structure as m
assert os.path.normcase(m.__file__).startswith(os.path.normcase(tree)), m.__file__
import reversal_shadow  # noqa: F401  (wraps the tree's MarketStructure)
sys.argv = ["run_replay"]
runpy.run_module("engine_v2.run_replay", run_name="__main__")
