"""PLAN_Q (the zone-proximity scan reads a POI exactly like a KL zone) — mutants for `mutant_runner.py` (full suite
each; PLAN_Q §8).

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \
        engine_v2/plans/plan_q_inputs/mutants_plan_q.py --out <a short %TEMP% dir>

Every one must be KILLED. History: the first run (the drafted silent start gate, 2026-10-14) killed 10 / 11 — Q08
SURVIVED: the plan had called it equivalent, but the IC can lie past the establishment (the fib's CTS extends until
its confirmation), so the term binds and no test pinned it — pinned since in
`test_poi_activation_floor_idx_takes_the_max_of_its_terms`. The plan review (finding 2) turned the gate into an
assert; this list targets the as-built code (the asserts, their boundaries, the two hovers — review finding 3).
"""
ZP = "engine_v2/zones/zone_proximity.py"
PZ = "engine_v2/zones/poi_zones.py"
H1 = "engine_v2/charting/export_plotly.py"
M15 = "engine_v2/charting/export_m15_chart.py"

_FLOOR_ASSERT = "assert poi_floor_idx <= scan_start, ("
_END_ASSERT = "assert poi_end_idx is None or int(poi_end_idx) > scan_end, ("
_SWEEP_CALL = "pattern_confirmed_idx=pattern_confirmed_idx, lifecycle_floor_idx=lifecycle_floor_idx,"
_HOVER = "return int(poi_unclamped_activation_floor_idx(best))"

MUT = [
    # the scan's POI read
    ("Q01_gate_restored_to_the_activation_history", ZP, "for pz in cycle_poi_zones:",
     'for pz in [p for p in cycle_poi_zones if __import__("engine_v2.zones.poi_lifecycle", '
     'fromlist=["poi_active_as_of"]).poi_active_as_of(p, idx)]:'),
    ("Q02_floor_assert_removed", ZP, _FLOOR_ASSERT, "assert True, ("),
    ("Q07_floor_assert_boundary_exclusive", ZP, _FLOOR_ASSERT, "assert poi_floor_idx < scan_start, ("),
    ("Q05_floor_read_from_the_activation_history", ZP, "lifecycle_floor_idx=None,",
     'lifecycle_floor_idx=(pz.meta["activation_history"][0]["idx"] if pz.meta["activation_history"] else 10**9),'),
    ("Q10_floor_from_the_lossy_scalar", ZP, 'int(pz.meta["cts_established_idx"]), int(pz.ic_idx),',
     'int(pz.meta["confirmed_idx"] or 0), int(pz.ic_idx),'),
    ("Q06_end_assert_removed", ZP, _END_ASSERT, "assert True, ("),
    ("Q11_end_assert_boundary_inclusive", ZP, _END_ASSERT,
     "assert poi_end_idx is None or int(poi_end_idx) >= scan_end, ("),
    # the shared floor helper
    ("Q03_floor_drops_the_pattern_term", PZ, "floor = max(floor, int(pattern_confirmed_idx))", "floor = floor"),
    ("Q04_floor_drops_the_lifecycle_term", PZ, "floor = max(floor, int(lifecycle_floor_idx))", "floor = floor"),
    ("Q08_floor_drops_the_ic_term", PZ, "floor = max(int(cts_established_idx), int(ic_idx))",
     "floor = int(cts_established_idx)"),
    # the activation sweep's call site (the extraction must keep every term)
    ("Q12_sweep_call_drops_the_lifecycle_floor", PZ, _SWEEP_CALL,
     "pattern_confirmed_idx=pattern_confirmed_idx, lifecycle_floor_idx=None,"),
    ("Q13_sweep_call_drops_the_pattern_term", PZ, _SWEEP_CALL,
     "pattern_confirmed_idx=None, lifecycle_floor_idx=lifecycle_floor_idx,"),
    # the two trigger hovers
    ("H01_h1_hover_shows_the_establishment", H1, _HOVER, 'return int(best.meta["cts_established_idx"])'),
    ("H02_m15_hover_shows_the_establishment", M15, _HOVER, 'return int(best.meta["cts_established_idx"])'),
]
