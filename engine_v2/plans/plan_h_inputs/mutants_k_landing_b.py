"""PLAN_K landing B (the IC reference condition applied + the `no_ic_candidate` reason) — mutants for
`mutant_runner.py` (full suite each).

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \
        engine_v2/plans/plan_h_inputs/mutants_k_landing_b.py --jobs 9 --out <a short %TEMP% dir>

Every one must be KILLED.
"""
PZ = "engine_v2/zones/poi_zones.py"

MUT = [
    # the IC-candidate predicate (both consumers)
    ("B01_predicate_ignores_reference", PZ,
     "    return bool(touches_zone) and ic_ref_reason is None", "    return bool(touches_zone)"),
    ("B02_predicate_ignores_touch", PZ,
     "    return bool(touches_zone) and ic_ref_reason is None", "    return ic_ref_reason is None"),
    # the pick's call site
    ("B03_pick_unfiltered", PZ,
     "    eligible = {i: is_ic_candidate(touches[i], ref_reasons[i]) for i in candidates}",
     "    eligible = {i: is_ic_candidate(touches[i], None) for i in candidates}"),
    ("B04_patterns_ignore_eligibility", PZ,
     "    patterns = [c.ip for c in cands if eligible[c.ip.ic_idx] and c.ip.rank is not None]",
     "    patterns = [c.ip for c in cands if c.ip.rank is not None]"),
    # the none reason
    ("B05_reason_all_not_any", PZ,
     "reason=(\"no_pattern\" if any(is_ic_candidate(c.touches_zone, c.ic_ref_reason) for c in sel.candidates)",
     "reason=(\"no_pattern\" if all(is_ic_candidate(c.touches_zone, c.ic_ref_reason) for c in sel.candidates)"),
    ("B06_reason_touch_only", PZ,
     "reason=(\"no_pattern\" if any(is_ic_candidate(c.touches_zone, c.ic_ref_reason) for c in sel.candidates)",
     "reason=(\"no_pattern\" if any(c.touches_zone for c in sel.candidates)"),
    ("B07_reason_any_candidate", PZ,
     "reason=(\"no_pattern\" if any(is_ic_candidate(c.touches_zone, c.ic_ref_reason) for c in sel.candidates)",
     "reason=(\"no_pattern\" if sel.candidates"),
    ("B08_reason_labels_swapped", PZ,
     "                        else \"no_ic_candidate\"),", "                        else \"no_pattern\"),"),
]
