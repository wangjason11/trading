"""Mutants of Post-E·5 (PLAN_E §9.6) for `mutant_runner.py`: fib `cycle1_bos_idx` -> `cycle1_bos_anchor_idx`, KL
`pb_reconfirm_idx` -> `reconfirmed_idx`, the KL `expanded_last_*` trio deleted (== `bounds_steps[-1]`).

Not only revert-shaped (the 2026-09-29d lesson: the Post-E·4 review's value / swap / extra-key / exporter-alias mutants
all survived my revert-shaped ones): F = fib, K = KL, L = the mirror lists, X = an exporter alias.
The landing review's 10 (R1, M1, K14-K17, F7, L5, X2, X3; 6 SURVIVED at `896ffd4`) are appended - all must die after
the fold-in (exporter test, two-step mirror pin, second-expansion + key-set pins, Scenario-3 asserts).
Template: `(id, relpath, old, new)`; `old` must occur exactly once in the file.
"""
FT = "engine_v2/zones/fib_tracker.py"
KL = "engine_v2/zones/kl_zones_v1.py"
EDM = "engine_v2/multitf/entity_df_mutation.py"
EXP = "engine_v2/debug/export_fib_lifecycle.py"
EXZ = "engine_v2/debug/export_zones.py"

_W = '"cycle1_bos_anchor_idx": bos_idx,'
_R = 'cycle1_bos_anchor_idx = new_cross_fib.meta["cycle1_bos_anchor_idx"]'
_E = '"expanded": True,'
_C = '"reconfirmed_idx": int(ef.event_moment(ev)),'  # the fold-in's NIT 8 read

MUT = [
    # --- fib: revert, extra key, BOS_0 for BOS_1 (swap), the moment for the anchor, the reader's source ---
    ("F1_writer_revert", FT, _W, '"cycle1_bos_idx": bos_idx,'),
    ("F2_writer_alias", FT, _W, _W + ' "cycle1_bos_idx": bos_idx,'),
    ("F3_swap_bos0_anchor", FT, _W, '"cycle1_bos_anchor_idx": anchor_bos_idx,'),
    ("F4_moment_for_anchor", FT, _W, '"cycle1_bos_anchor_idx": cts_established_idx,'),
    ("F5_reader_bos_by_cycle", FT, _R, "cycle1_bos_anchor_idx = self._bos_by_cycle[(sid, 1)][0]"),
    ("F6_reader_get_fallback", FT, _R,
     'cycle1_bos_anchor_idx = new_cross_fib.meta.get("cycle1_bos_anchor_idx", new_cross_fib.bos_idx)'),
    # --- KL expansion: an extra (index / non-index) key back, the flag dropped, the step's own values wrong ---
    ("K1_extra_expanded_last_idx", KL, _E, _E + ' "expanded_last_idx": int(ev.idx),'),
    ("K2_extra_expanded_last_price", KL, _E, _E + ' "expanded_last_price": float(price),'),
    ("K3_extra_expanded_last_event", KL, _E, _E + ' "expanded_last_event": str(ev.type),'),
    ("K4_drop_expanded_flag", KL, _E, ""),
    ("K5_step_price_prev", KL, '"price": float(price),', '"price": float((ev.meta or {}).get("prev", price)),'),
    ("K6_step_start_plus1", KL, '"start_idx": int(ev.idx),       # expansion happens HERE',
     '"start_idx": int(ev.idx) + 1,       # expansion happens HERE'),
    ("K7_step_event_const", KL, '"event": str(ev.type),', '"event": "THRESHOLD_UPDATED",'),
    # --- KL reconfirm: revert, alias, the proximity moment / the anchor for the moment, confirmed_idx overwritten,
    #     the CTS-kind filter dropped (the cycle's BOS zone upgraded instead) ---
    ("K8_reconfirm_revert", KL, _C, '"pb_reconfirm_idx": int(ev.idx),'),
    ("K9_reconfirm_alias", KL, _C, _C + ' "pb_reconfirm_idx": int(ev.idx),'),
    ("K10_swap_confirmed_idx", KL, _C, '"reconfirmed_idx": int(zmeta["confirmed_idx"]),'),
    ("K11_anchor_for_moment", KL, _C, '"reconfirmed_idx": int(ef.cts_anchor_idx(ev)),'),
    ("K12_overwrite_confirmed_idx", KL, _C, _C + ' "confirmed_idx": int(ev.idx),'),
    ("K13_any_kind_zone", KL, 'if z.source_kind != "CTS":', 'if False:'),
    # --- mirror lists: a key dropped, the old name kept / listed ---
    ("L1_drop_cycle1_bos_anchor", EDM, '    "cycle1_bos_anchor_idx",  # the cycle-1',
     '    # "cycle1_bos_anchor_idx",  # the cycle-1'),
    ("L2_list_old_fib_name", EDM, '    "cycle1_bos_anchor_idx",  # the cycle-1', '    "cycle1_bos_idx",  # the cycle-1'),
    ("L3_drop_reconfirmed", EDM, '    "reconfirmed_idx",      # KL CTS zone', '    # "reconfirmed_idx",      # KL CTS zone'),
    ("L4_dead_expanded_last_entry", EDM, '    "bos_anchor_idx",       # POI (the owning',
     '    "expanded_last_idx", "bos_anchor_idx",       # POI (the owning'),
    # --- an alias added in an exporter (no exporter test exists: the Post-E·4 review's R1 survived) ---
    ("X1_fib_exporter_alias", EXP, '"meta": m,',
     '"meta": {**m, "cycle1_bos_idx": m["cycle1_bos_anchor_idx"]} if "cycle1_bos_anchor_idx" in m else m,'),
    # --- the landing review's mutants (2026-09-30) ---
    # reader: cond2's window starts at the cross fib's OWN bos_idx (BOS_0's anchor) instead of the meta key
    ("R1_reader_fib_bos0", FT, "c1_start = min(cycle1_bos_anchor_idx, cts_idx)",
     "c1_start = min(new_cross_fib.bos_idx, cts_idx)"),
    # mirror shift: only the FIRST bounds_steps entry (INIT) is shifted; every expansion step stays slice-local
    # (bounds_steps[-1] is now the ONLY home of the last expansion's moment)
    ("M1_mirror_shift_init_step_only", EDM, 'if isinstance(new_step.get("start_idx"), int):',
     'if isinstance(new_step.get("start_idx"), int) and not steps:'),
    # KL writer ordering: a new step is inserted after INIT instead of appended -> with >=2 expansions
    # bounds_steps[-1] is the FIRST expansion, not the last (identical with 1 expansion)
    ("K14_step_insert_not_append", KL, "steps.append({", "steps.insert(1, {"),
    # extra-key: a duplicate of bounds_steps[-1] re-added under a NEW non-index name
    ("K15_dup_new_name_nonidx", KL, _E, _E + ' "last_expansion_price": float(price),'),
    # reconfirm: the upgrade matched on sid only (the cycle filter dropped) -> the sid's FIRST CTS zone upgraded
    ("K16_reconfirm_ignore_cycle", KL, 'if zmeta.get("cycle_id") != ev_cycle:', "if False:"),
    # reconfirm: the method is not upgraded (only the moment recorded)
    ("K17_reconfirm_method_kept", KL, '"confirmation_method": "pullback",',
     '"confirmation_method": zmeta.get("confirmation_method"),'),
    # fib second writer path: the Scenario-3 single also carries the cross-only key
    ("F7_scenario3_writes_key", FT, 'meta={"activated_at": cts_established_idx, "scenario": 3},',
     'meta={"activated_at": cts_established_idx, "scenario": 3, "cycle1_bos_anchor_idx": bos_idx},'),
    # list misfile: reconfirmed_idx ALSO in the EVENT list (the pre-Post-E.2 misfile shape of pb_reconfirm_idx)
    ("L5_misfile_reconfirmed_in_event_list", EDM, '    "ended_watch_pattern_anchor_idx",  # BOS_CONFIRMED',
     '    "reconfirmed_idx", "ended_watch_pattern_anchor_idx",  # BOS_CONFIRMED'),
    # exporter alias built from a computed string (no single constant holds the old key; no exporter test)
    ("X2_kl_exporter_computed_alias", EXZ, '"meta": z.meta,',
     '"meta": {**z.meta, **({"pb_reconfirm" + "_idx": z.meta["reconfirmed_idx"]} if "reconfirmed_idx" in z.meta else {})},'),
    # exporter drops bounds_steps from the KL CSV (the last expansion now lives ONLY there)
    ("X3_kl_exporter_drops_bounds_steps", EXZ, '"meta": z.meta,',
     '"meta": {k: v for k, v in (z.meta or {}).items() if k != "bounds_steps"},'),
]
