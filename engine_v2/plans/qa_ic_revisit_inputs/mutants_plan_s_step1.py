"""PLAN_S step 1 mutants (§2.4): every fill-threshold site set back to a literal 0.70, the constant itself, and the two
tracker configs' wiring — each must fail the suite.

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py engine_v2/plans/qa_ic_revisit_inputs/mutants_plan_s_step1.py
"""
C = "IMBALANCE_FILL_THRESHOLD"
MUT = [
    ("s1_const", "engine_v2/common/types.py", f"{C} = 0.65\n", f"{C} = 0.70\n"),
    ("s1_is_filled", "engine_v2/common/types.py",
     f"        check_to_idx: int,\n        fill_threshold: float = {C},\n    ) -> bool:",
     "        check_to_idx: int,\n        fill_threshold: float = 0.70,\n    ) -> bool:"),
    ("s1_has_unfilled", "engine_v2/patterns/imbalance.py",
     f"    fill_threshold: float = {C},\n    *,\n    direction: Optional[int] = None,\n    evaluated_at: Optional[int],\n) -> bool:",
     "    fill_threshold: float = 0.70,\n    *,\n    direction: Optional[int] = None,\n    evaluated_at: Optional[int],\n) -> bool:"),
    ("s1_iter_unfilled", "engine_v2/patterns/imbalance.py",
     f"    fill_threshold: float = {C},\n    *,\n    direction: Optional[int] = None,\n    evaluated_at: Optional[int],\n) -> Iterator",
     "    fill_threshold: float = 0.70,\n    *,\n    direction: Optional[int] = None,\n    evaluated_at: Optional[int],\n) -> Iterator"),
    ("s1_cross_cycle", "engine_v2/zones/cross_cycle_fib.py",
     f"    fill_threshold: float = {C},\n", "    fill_threshold: float = 0.70,\n"),
    ("s1_fib_config", "engine_v2/zones/fib_tracker.py",
     f"    fill_threshold: float = {C}\n", "    fill_threshold: float = 0.70\n"),
    ("s1_poi_config", "engine_v2/zones/poi_zones.py",
     f"    fill_threshold: float = {C}  #", "    fill_threshold: float = 0.70  #"),
    ("s1_orch_tracker", "engine_v2/pipeline/orchestrator.py",
     f"            fill_threshold={C},     # ==", "            fill_threshold=0.70,     # =="),
    ("s1_ms_tracker", "engine_v2/structure/structure_engine.py",
     f"fill_threshold={C}),", "fill_threshold=0.70),"),
]
