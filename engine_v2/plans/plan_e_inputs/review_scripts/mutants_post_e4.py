"""Mutants of the Post-E·4 meta-key renames (PLAN_E §9.5) for `mutant_runner.py` — all KILLED at `ed99914`.

M1-M12: my pre-review loop (an old key restored at each emit site, each new key dropped from its shift list, the debug
print back on `.get`, an alias). R2-R4: the landing review's in-code survivors, killed by the fold-in (a wrong value, a
swapped pair, an extra key inside the cycle >= 1 BOS wrapper-call literal). The review's R1 (an alias added in an
exporter) has no test by design — `/compare` with `cmp_meta_rename.py` guards the export layer.
Template: `(id, relpath, old, new)`; `old` must occur exactly once in the file.
"""
MS = "engine_v2/structure/market_structure.py"
POI = "engine_v2/zones/poi_zones.py"
EDM = "engine_v2/multitf/entity_df_mutation.py"

MUT = [
    ("M1_bos_c1_pb_start", MS,
     '                            "source": "pullback_extreme",\n                            "confirmed_at": int(apply_idx),\n'
     '                            "last_pullback_apply_idx":',
     '                            "source": "pullback_extreme",\n                            "confirmed_at": int(apply_idx),\n'
     '                            "pb_start":'),
    ("M2_bos_c0_pb_start", MS,
     '                            "source": "initial_prior_extreme",\n                            "confirmed_at": int(apply_idx),\n'
     '                            "last_pullback_apply_idx":',
     '                            "source": "initial_prior_extreme",\n                            "confirmed_at": int(apply_idx),\n'
     '                            "pb_start":'),
    ("M3_range_offline_cts_idx", MS,
     '"cts_anchor_idx": None if st.cts is None else st.cts.idx,', '"cts_idx": None if st.cts is None else st.cts.idx,'),
    ("M4_range_pullback_cts_idx", MS,
     '                        "cts_anchor_idx": st.cts.idx,\n', '                        "cts_idx": st.cts.idx,\n'),
    ("M5_range_proximity_cts_idx", MS,
     '                        "cts_anchor_idx": int(st.cts.idx),\n', '                        "cts_idx": int(st.cts.idx),\n'),
    ("M6_poi_bos_idx", POI, '"bos_anchor_idx": fib_state.bos_idx,', '"bos_idx": fib_state.bos_idx,'),
    ("M7_poi_cts_idx", POI, '"cts_anchor_idx": fib_state.cts_idx,', '"cts_idx": fib_state.cts_idx,'),
    ("M8_list_drop_last_pullback", EDM,
     '    "last_pullback_apply_idx",  # BOS_CONFIRMED', '    # "last_pullback_apply_idx",  # BOS_CONFIRMED'),
    ("M9_zone_list_drop_bos_anchor", EDM, '    "bos_anchor_idx",       # POI', '    # "bos_anchor_idx",       # POI'),
    ("M10_zone_list_drop_cts_anchor", EDM, '    "cts_anchor_idx",       # POI', '    # "cts_anchor_idx",       # POI'),
    ("M11_debug_print_get", POI,
     "bos_anchor_idx={z.meta['bos_anchor_idx']}", "bos_anchor_idx={z.meta.get('bos_idx')}"),
    ("M12_alias_both_keys_poi", POI,
     '"bos_anchor_idx": fib_state.bos_idx,', '"bos_anchor_idx": fib_state.bos_idx, "bos_idx": fib_state.bos_idx,'),
    ("R2_prox_value_candle", MS,
     '                        "cts_anchor_idx": int(st.cts.idx),\n', '                        "cts_anchor_idx": int(candle_idx),\n'),
    ("R3_poi_swap", POI, '"bos_anchor_idx": fib_state.bos_idx,', '"bos_anchor_idx": fib_state.cts_idx,'),
    ("R4_extra_cts_idx_bos", MS,
     '                            "source": "pullback_extreme",\n',
     '                            "source": "pullback_extreme",\n                            "cts_idx": int(self.state.cts.idx),\n'),
]
