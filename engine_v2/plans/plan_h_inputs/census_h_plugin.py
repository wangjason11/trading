"""PLAN_H §8.4 — the test census (PR7; `plan_e_inputs/review_scripts/census_plugin.py`'s recipe): per test, what the
POI layer computed — every `derive_poi_zones` POI (sid, cycle, IC, confirmed_idx, status, activation history), every
direct `_compute_poi_activation_history` result, every MS resolver answer (`compute_poi_inners_for_cycle`) and every
`select_ip_ic` pick — plus the test's outcome.

    PYTHONPATH="TREE;engine_v2/plans/plan_h_inputs" CENSUS_OUT=out.json \
        python -m pytest TREE/engine_v2/tests -q -p no:cacheprovider -p census_h_plugin --deselect <smoke>

Run once on A's tree and once on the V3 tree; `census_h_diff.py` lists every test whose observations or outcome
moved (→ re-pin / re-derive / retire in landing B). The wrappers are installed at plugin import, before the test
modules import the names; nothing is changed but the recording.
"""
import json
import os
import sys

import engine_v2.zones.poi_zones as pz
import engine_v2.structure.structure_engine as se
import engine_v2.pipeline.orchestrator as orch

REC = {}
CUR = ["<collect>"]


def _bucket():
    return REC.setdefault(CUR[0], {"poi": [], "act": [], "res": [], "sel": [], "outcome": {}})


def _hist(h):
    return [[int(e["idx"]), bool(e["active"]), e.get("reason"), list(e.get("versions", []))] for e in h]


_real_derive = pz.derive_poi_zones
_real_act = pz._compute_poi_activation_history
_real_res = pz.compute_poi_inners_for_cycle
_real_sel = pz.select_ip_ic
_depth = [0]


def derive_poi_zones(*a, **k):
    _depth[0] += 1
    try:
        zones = _real_derive(*a, **k)
    finally:
        _depth[0] -= 1
    _bucket()["poi"].append([
        [int(z.meta["structure_id"]), int(z.meta["cycle_id"]), int(z.ic_idx), z.meta.get("confirmed_idx"),
         z.meta.get("status"), list(z.meta.get("versions", [])), _hist(z.meta.get("activation_history", []))]
        for z in zones])
    return zones


def _compute_poi_activation_history(*a, **k):
    h = _real_act(*a, **k)
    if _depth[0] == 0:  # a direct (test) call; the pipeline's are inside derive_poi_zones
        _bucket()["act"].append([int(k.get("ic_idx", -1)), _hist(h)])
    return h


def compute_poi_inners_for_cycle(*a, **k):
    out = _real_res(*a, **k)
    sid = a[6] if len(a) > 6 else k.get("structure_id", 0)
    cyc = a[7] if len(a) > 7 else k.get("cycle_id", 0)
    _bucket()["res"].append([int(sid), int(cyc), int(a[1]), int(a[3]), [round(float(x), 6) for x in out]])
    return out


def select_ip_ic(df, fib_state, config, **kw):
    sel = _real_sel(df, fib_state, config, **kw)
    _bucket()["sel"].append([int(fib_state.structure_id), int(fib_state.cycle_id), int(fib_state.bos_idx),
                             int(fib_state.cts_idx), None if sel.pick is None else int(sel.pick.ic_idx),
                             None if sel.pick is None else sel.pick.ip_rank, bool(kw.get("classify_misses"))])
    return sel


pz.derive_poi_zones = derive_poi_zones
pz._compute_poi_activation_history = _compute_poi_activation_history
pz.compute_poi_inners_for_cycle = compute_poi_inners_for_cycle
pz.select_ip_ic = select_ip_ic
se.compute_poi_inners_for_cycle = compute_poi_inners_for_cycle
orch.derive_poi_zones = derive_poi_zones


def pytest_runtest_setup(item):
    CUR[0] = item.nodeid


def pytest_runtest_logreport(report):
    b = REC.setdefault(report.nodeid, {"poi": [], "act": [], "res": [], "sel": [], "outcome": {}})
    b["outcome"][report.when] = report.outcome


def pytest_sessionfinish(session, exitstatus):
    out = os.environ.get("CENSUS_OUT", "census_h.json")
    with open(out, "w") as fh:
        json.dump(REC, fh, default=str)
    print(f"\n[census_h] {len(REC)} tests -> {out}", file=sys.stderr)
