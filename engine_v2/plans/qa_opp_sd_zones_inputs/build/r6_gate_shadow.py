"""PLAN_R step 3's pool shadow (r5_shadow's method; its `full` variant reproduced step 2's pool byte for byte): the
pre-overlay pipeline at `77f8ccf` (step 1 + the 0a bound, = S_Q) with the (0,1) / (1,2) proximity lists replaced by the
step-3 gate simulator's chains (`m9_gate_sim.py`): (0,1) 748 sd:POI, 820 opp_sd on sub_3 c2 POI 0.58096, 872 sd:BOS
0.57494; (1,2) 935 sd:POI, 954 opp_sd on sub_6 c0 POI 0.57306, 1035 sd:POI — step 2's chains minus 1045 / 1046. Runs the
FULL replay (CSVs + charts into the worktree's artifacts/). Run from the worktree root, PYTHONPATH = the worktree."""
import sys

sys.argv = [sys.argv[0]]
import engine_v2.run_replay as rr  # noqa: E402
import engine_v2.pipeline.orchestrator as orch  # noqa: E402
from engine_v2.zones.zone_proximity import ZoneProximityTrigger, check_zone_proximity as _orig  # noqa: E402


def mk(sid, cyc, d, i, inner, k):
    return ZoneProximityTrigger(structure_id=sid, cycle_id=cyc, direction=d, idx=i, trigger_inner=inner, zone_kind=k,
                                proximity_pips=8, pip_size=0.0001, timeframe="H1", meta={"shadow": True})


def patched(**kw):
    out = _orig(**kw)
    l01, l12 = out[(0, 1)], out[(1, 2)]
    assert [t.idx for t in l01] == [748], [t.idx for t in l01]
    assert [t.idx for t in l12] == [935, 954, 1035], [t.idx for t in l12]
    out[(0, 1)] = [l01[0], mk(0, 1, "opp_sd", 820, 0.58096, "POI"), mk(0, 1, "sd", 872, 0.57494, "BOS")]
    out[(1, 2)] = [l12[0], mk(1, 2, "opp_sd", 954, 0.57306, "POI"), l12[2]]
    for k in ((0, 1), (1, 2)):
        print(f"[shadow] {k} triggers:", [(t.idx, t.direction, t.zone_kind, t.trigger_inner) for t in out[k]])
    return out


orch.check_zone_proximity = patched
rr.main()
