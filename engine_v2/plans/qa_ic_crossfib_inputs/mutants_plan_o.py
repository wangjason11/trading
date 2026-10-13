"""PLAN_O (the IC reference's CTS = cycle T−1's for every fib record) — mutants for `mutant_runner.py` (full suite each;
PLAN_O §7 + the plan review's X1 / X2 / full revert / assert deletion).

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \
        engine_v2/plans/qa_ic_crossfib_inputs/mutants_plan_o.py --out <a short %TEMP% dir>

Every one must be KILLED except O16 (EQUIVALENT: the both-or-neither assert makes `anchor` and `known` None together).
"""
FT = "engine_v2/zones/fib_tracker.py"
ZR = "engine_v2/charting/_zone_render.py"

_BLOCK_NEW = (
    "        if target >= 1:\n"
    "            cts = self._cts_by_cycle.get((sid, target - 1))\n"
)
_ASSERT2 = ("assert (anchor is None) == (known is None), (\n"
            "                f\"[fib] sid={sid} cycle={target}: establishment pattern anchor {anchor} vs moment {known}\")")
_HEAD_BLOCK = (   # the builder as at caf0f01 (the start cycle's CTS, `[...]` reads)
    "        start_cycle = int(fib_state.meta[\"cross_start_cycle\"]) if fib_state.meta.get(\"cross_cycle\") else target\n"
    "        if start_cycle >= 1:\n"
    "            cts = self._cts_by_cycle.get((sid, start_cycle - 1))\n"
    "            assert cts is not None\n"
    "            anchor = self._est_pattern_anchor[(sid, start_cycle)]\n"
    "            known = self._bos_moment_by_cycle[(sid, start_cycle)]\n"
    "            brk = (anchor, known) if (evaluated_at is None or known <= int(evaluated_at)) else None\n"
    "            refs.append((False, float(cts[1]), \"prev_cycle_cts\", brk))\n"
    "        if False:\n"
    "            cts = self._cts_by_cycle.get((sid, target - 1))\n"
)
_HOVER = 'cycle = f", cycle {int(fib_st.cycle_id) - 1}"'

MUT = [
    # the builder
    ("O01_cts_of_target_not_t_minus_1", FT, "cts = self._cts_by_cycle.get((sid, target - 1))",
     "cts = self._cts_by_cycle.get((sid, target))"),
    ("O02_cycle_1_excluded", FT, "if target >= 1:", "if target > 1:"),
    ("O03_anchor_of_t_minus_1", FT, "anchor = self._est_pattern_anchor.get((sid, target))",
     "anchor = self._est_pattern_anchor.get((sid, target - 1))"),
    ("O04_moment_of_t_minus_1", FT, "known = self._bos_moment_by_cycle.get((sid, target))",
     "known = self._bos_moment_by_cycle.get((sid, target - 1))"),
    ("O05_no_anchor_guard", FT, "if (anchor is not None and (evaluated_at is None or known <= int(evaluated_at)))",
     "if ((evaluated_at is None or known <= int(evaluated_at)))"),
    ("O06_known_strict", FT, "known <= int(evaluated_at)))", "known < int(evaluated_at)))"),
    ("O07_anchor_indexed", FT, "anchor = self._est_pattern_anchor.get((sid, target))",
     "anchor = self._est_pattern_anchor[(sid, target)]"),
    ("O08_brk_swapped", FT, "brk = (anchor, known) if (anchor", "brk = (known, anchor) if (anchor"),
    ("O09_cross_from_0_excluded", FT, "if target >= 1:",
     "if target >= 1 and not (fib_state.meta.get(\"cross_cycle\") and int(fib_state.meta[\"cross_start_cycle\"]) == 0):"),
    ("O10_X1_cts_by_start_cycle", FT, "cts = self._cts_by_cycle.get((sid, target - 1))",
     "cts = self._cts_by_cycle.get((sid, int(fib_state.meta[\"cross_start_cycle\"]) if fib_state.meta.get(\"cross_cycle\")"
     " else target - 1))"),
    ("O11_builder_reverted_to_head", FT, _BLOCK_NEW, _HEAD_BLOCK),
    ("O12_assert_deleted", FT, _ASSERT2, "pass"),
    ("O13_assert_neutralized", FT, "assert (anchor is None) == (known is None), (",
     "assert True or (anchor is None) == (known is None), ("),
    # the hover
    ("O14_hover_no_minus_1", ZR, _HOVER, 'cycle = f", cycle {int(fib_st.cycle_id)}"'),
    ("O15_hover_X2_prints_start", ZR, _HOVER,
     "cycle = f\", cycle {int(fib_st.meta['cross_start_cycle']) if fib_st.meta.get('cross_cycle') "
     "else int(fib_st.cycle_id) - 1}\""),
    ("O17_hover_back_to_start_minus_1", ZR, _HOVER,
     "cycle = f\", cycle {(int(fib_st.meta['cross_start_cycle']) if fib_st.meta.get('cross_cycle') "
     "else int(fib_st.cycle_id)) - 1}\""),
    # EQUIVALENT (expected to survive)
    ("O16_EQUIV_known_guard", FT, "if (anchor is not None and (evaluated_at", "if (known is not None and (evaluated_at"),
]
