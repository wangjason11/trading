"""PLAN_I session 2 — the REAL replay through a variant tree (`variant_patch_i.py`), PLAN_H §8.4's pattern.

    cd WORKDIR && python REPO/engine_v2/plans/plan_i_inputs/replay_variant_i.py TREE LOG.json

WORKDIR holds a copy of `oanda.cfg`; the replay writes WORKDIR/artifacts/ (the main checkout is never touched, so
variants run side by side). TREE/engine_v2 is put first on sys.path (asserted). At exit the variant's
`engine_v2.multitf.cross_fib_limit.LOG` (V2+) is dumped to LOG.json.
"""
import atexit
import json
import os
import runpy
import sys

tree = os.path.abspath(sys.argv[1])
log_path = os.path.abspath(sys.argv[2])
sys.path.insert(0, tree)
import engine_v2.zones.fib_tracker as _ft  # noqa: E402
assert os.path.normcase(_ft.__file__).startswith(os.path.normcase(tree)), _ft.__file__


@atexit.register
def _dump():
    try:
        from engine_v2.multitf.cross_fib_limit import LOG
    except ImportError:
        LOG = []
    with open(log_path, "w") as fh:
        json.dump(LOG, fh, default=str)
    print(f"[replay_variant_i] {len(LOG)} log records -> {log_path}")


os.makedirs("artifacts/debug", exist_ok=True)
os.makedirs("artifacts/charts", exist_ok=True)
sys.argv = ["run_replay"]
runpy.run_module("engine_v2.run_replay", run_name="__main__")
