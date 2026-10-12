"""PLAN_M landing 1 (the live-`P` plumbing) — mutants for `mutant_runner.py` (full suite each; PLAN_M §9 L1's list +
the call sites).

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \
        engine_v2/plans/plan_m_inputs/mutants_m1.py --jobs 14 --out <a short %TEMP% dir>

Every one must be KILLED, except M12b (stated EQUIVALENT: the stamp only RE-ORDERS the limit keys, and its tail list
read at None is a superset of the one at the cap — every key the finalize wrote is still moved to the tail).
"""
CF = "engine_v2/zones/cross_cycle_fib.py"
FT = "engine_v2/zones/fib_tracker.py"
ZR = "engine_v2/charting/_zone_render.py"

MUT = [
    # CrossFibLimit
    ("M01_price_at_exclusive", CF,
     "            if int(known_at) > int(moment):", "            if int(known_at) >= int(moment):"),
    ("M02_price_at_none_reads_start", CF,
     "        if moment is None or not self.price_moves:\n            return self.price\n",
     "        if not self.price_moves:\n            return self.price\n        if moment is None:\n"
     "            return self.price_at_start\n"),
    ("M03_localized_leaves_the_moves", CF,
     "        if self.price_moves:\n            changes[\"price_moves\"] = tuple((int(k) - int(offset), p) for k, p in self.price_moves)\n",
     ""),
    ("M04_localized_always_replaces", CF,
     "        return replace(self, **changes) if changes else self", "        return replace(self, **changes)"),
    ("M05_moves_until_ignores_moment", CF,
     "        return tuple(mv for mv in self.price_moves if int(mv[0]) <= int(moment))",
     "        return self.price_moves"),
    # FibTracker — the live read and the fix
    ("M06_cross_allowed_reads_last_p", FT,
     "        return self._limit_reaches(int(sid), int(target_cycle), self._evaluated_at)",
     "        return self._limit_reaches(int(sid), int(target_cycle), None)"),
    ("M07_cross_allowed_caches_first_read", FT,
     "        return self._limit_reaches(int(sid), int(target_cycle), self._evaluated_at)",
     "        cache = self.__dict__.setdefault(\"_m07\", {})\n"
     "        if (int(sid), int(target_cycle)) not in cache:\n"
     "            cache[(int(sid), int(target_cycle))] = self._limit_reaches(int(sid), int(target_cycle), "
     "self._evaluated_at)\n"
     "        return cache[(int(sid), int(target_cycle))]"),
    ("M08_cross_allowed_ignores_the_fix", FT,
     "        fixed = self._cross_permission.get((int(sid), int(target_cycle)))\n        if fixed is not None:\n"
     "            return fixed\n        if lim.price_moves",
     "        if lim.price_moves"),
    ("M09_fix_never_stored", FT,
     "        self._cross_permission[key] = allowed\n", ""),
    ("M10_second_fix_keeps_the_first", FT,
     "        if key in self._cross_permission:\n            raise CrossFibLimitError(",
     "        if key in self._cross_permission:\n            return self._cross_permission[key]\n"
     "            raise CrossFibLimitError("),
    ("M11_establishment_reads_without_fixing", FT,
     "        if cycle_id == 0 or not self._fix_cross_permission(sid, cycle_id):",
     "        if cycle_id == 0 or not self._cross_allowed(sid, cycle_id):"),
    ("M12_no_moment_no_raise", FT,
     "        if lim.price_moves and self._evaluated_at is None:", "        if False:"),
    # the structure reference + the watch
    ("M13_reference_ignores_the_fix", FT,
     "        reaches = fixed if fixed is not None else self._limit_reaches(int(sid), int(target_cycle), at)",
     "        reaches = self._limit_reaches(int(sid), int(target_cycle), at)"),
    ("M14_ic_reference_drops_the_moment", FT,
     "        lim = self._structure_reference_limit(sid, target, evaluated_at)",
     "        lim = self._structure_reference_limit(sid, target)"),
    ("M15_watch_reads_last_p", FT,
     "                brk = find_cross_limit_break(df, float(lim.price_at(evaluated_at)), sd, start, known_at=evaluated_at)",
     "                brk = find_cross_limit_break(df, float(lim.price), sd, start, known_at=evaluated_at)"),
    # the export
    ("M16_export_price_is_last_p", FT,
     "            \"cross_fib_limit_price\": lim.price_at(at),", "            \"cross_fib_limit_price\": lim.price,"),
    ("M17_export_m_at_last_p", FT,
     "            \"cross_fib_limit_cycle\": self.cross_fib_limit_cycle(int(sid), at),",
     "            \"cross_fib_limit_cycle\": self.cross_fib_limit_cycle(int(sid)),"),
    ("M18_export_counts_every_move", FT,
     "        moves = lim.moves_until(at)", "        moves = lim.moves_until(None)"),
    ("M19_keys_always", FT,
     "        if moves:\n            out[\"cross_fib_limit_price_at_start\"]",
     "        if True:\n            out[\"cross_fib_limit_price_at_start\"]"),
    ("M20_keys_never", FT,
     "        if moves:\n            out[\"cross_fib_limit_price_at_start\"]",
     "        if False:\n            out[\"cross_fib_limit_price_at_start\"]"),
    ("M21_keys_first", FT,
     "        out: Dict[str, Any] = {\n            \"cross_fib_limit_price\": lim.price_at(at),",
     "        out: Dict[str, Any] = {}\n        if lim.moves_until(at):\n"
     "            out.update(cross_fib_limit_price_at_start=lim.price_at_start,\n"
     "                       cross_fib_limit_move_count=len(lim.moves_until(at)))\n"
     "        out = {**out,\n            \"cross_fib_limit_price\": lim.price_at(at),"),
    ("M22_start_key_holds_last_p", FT,
     "            out[\"cross_fib_limit_price_at_start\"] = lim.price_at_start",
     "            out[\"cross_fib_limit_price_at_start\"] = lim.price"),
    ("M23_count_ignores_the_cap", FT,
     "            out[\"cross_fib_limit_move_count\"] = len(moves)",
     "            out[\"cross_fib_limit_move_count\"] = len(lim.price_moves)"),
    ("M12a_finalize_drops_the_cap", FT,
     "            limit_meta = self._cross_fib_limit_meta(int(sid), lifecycle_cap)",
     "            limit_meta = self._cross_fib_limit_meta(int(sid))"),
    ("M12b_stamp_drops_the_cap", FT,                                # EQUIVALENT (see the module docstring)
     "            tail_keys = list(self._cross_fib_limit_meta(int(fib.structure_id), lifecycle_cap))",
     "            tail_keys = list(self._cross_fib_limit_meta(int(fib.structure_id)))"),
    # the hover
    ("M24_hover_suffix_dropped", ZR,
     "    if \"cross_fib_limit_move_count\" in m:\n        moved = (",
     "    if False:\n        moved = ("),
]
