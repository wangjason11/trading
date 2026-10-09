"""PLAN_J — the REAL replay through a variant tree (`variant_patch_j.py`), PLAN_I's `replay_variant_i.py` pattern.

    cd WORKDIR && python REPO/engine_v2/plans/plan_j_inputs/replay_variant_j.py TREE LOG.json

WORKDIR holds a copy of `oanda.cfg`; the replay writes WORKDIR/artifacts/ (the main checkout is never touched).
TREE/engine_v2 is put first on sys.path (asserted). At exit the variant's `market_structure.PJ_LOG` is dumped.
"""
import atexit
import json
import os
import runpy
import sys

tree = os.path.abspath(sys.argv[1])
log_path = os.path.abspath(sys.argv[2])
sys.path.insert(0, tree)
import engine_v2.structure.market_structure as _ms  # noqa: E402
assert os.path.normcase(_ms.__file__).startswith(os.path.normcase(tree)), _ms.__file__


@atexit.register
def _dump():
    log = getattr(_ms, "PJ_LOG", [])
    with open(log_path, "w") as fh:
        json.dump(log, fh, default=str)
    print(f"[replay_variant_j] {len(log)} log records -> {log_path}")


os.makedirs("artifacts/debug", exist_ok=True)
os.makedirs("artifacts/charts", exist_ok=True)
sys.argv = ["run_replay"]
runpy.run_module("engine_v2.run_replay", run_name="__main__")
