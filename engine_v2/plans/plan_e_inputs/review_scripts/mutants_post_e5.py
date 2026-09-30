"""Mutants of Post-E·5 (PLAN_E §9.6) for `mutant_runner.py`: fib `cycle1_bos_idx` -> `cycle1_bos_anchor_idx`, KL
`pb_reconfirm_idx` -> `reconfirmed_idx`, the KL `expanded_last_*` trio deleted (== `bounds_steps[-1]`).

Not only revert-shaped (the 2026-09-29d lesson: the Post-E·4 review's value / swap / extra-key / exporter-alias mutants
all survived my revert-shaped ones): F = fib, K = KL, L = the mirror lists, X = an exporter alias.
Template: `(id, relpath, old, new)`; `old` must occur exactly once in the file.
"""
FT = "engine_v2/zones/fib_tracker.py"
KL = "engine_v2/zones/kl_zones_v1.py"
EDM = "engine_v2/multitf/entity_df_mutation.py"
EXP = "engine_v2/debug/export_fib_lifecycle.py"

_W = '"cycle1_bos_anchor_idx": bos_idx,'
_R = 'cycle1_bos_anchor_idx = new_cross_fib.meta["cycle1_bos_anchor_idx"]'
_E = '"expanded": True,'
_C = '"reconfirmed_idx": int(ev.idx),'

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
]
