"""PLAN_K landing A's mutation loop (§6 "Mutation": every branch + the call sites; the full suite per mutant).

usage (repo root): python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \
    engine_v2/plans/plan_h_inputs/mutants_k_landing_a.py
Expected equivalent (noted in PLAN_K §6 / §11): W01 (the detector's own `check_break` tests the close), W11 (every
pattern spans >= 2 candles — a later anchor cannot confirm earlier; build deviation (3)).
Result 2026-10-09 (PLAN_K §11.1): first pass 77 / 89 KILLED; 10 gaps closed by tests (re-run 10 / 10 KILLED); final
87 KILLED + the 2 equivalent.
"""
W = "engine_v2/patterns/cross_limit_watch.py"
PZ = "engine_v2/zones/poi_zones.py"
FT = "engine_v2/zones/fib_tracker.py"
CCF = "engine_v2/zones/cross_cycle_fib.py"
SL = "engine_v2/zones/structure_lifecycle.py"
CFL = "engine_v2/multitf/cross_fib_limit.py"
EDM = "engine_v2/multitf/entity_df_mutation.py"
MS = "engine_v2/structure/market_structure.py"
EF = "engine_v2/structure/event_fields.py"
ORC = "engine_v2/pipeline/orchestrator.py"
ZR = "engine_v2/charting/_zone_render.py"
XZ = "engine_v2/debug/export_zones.py"

MUT = [
    # --- the watch
    ("W01_no_close_gate", W, "        if not (c > thr if sd == 1 else c < thr):\n            continue\n",
     "        if False:\n            continue\n"),
    ("W02_confirmation_ignored", W,
     "apply_idx = ev.confirmation_idx if ev.status == PatternStatus.CONFIRMED else ev.end_idx",
     "apply_idx = ev.end_idx"),
    ("W03_known_at_strict", W, "if known_at is not None and apply_idx > int(known_at):",
     "if known_at is not None and apply_idx >= int(known_at):"),
    ("W04_known_at_ignored", W, "if known_at is not None and apply_idx > int(known_at):",
     "if False and known_at is not None:"),
    ("W05_first_anchor_wins", W, "if best is None or (apply_idx, a) < (best[1], best[0]):", "if best is None:"),
    ("W06_tie_later_anchor", W, "if best is None or (apply_idx, a) < (best[1], best[0]):",
     "if best is None or (apply_idx, -a) < (best[1], -best[0]):"),
    ("W07_detector_unbounded", W, "bp = BreakoutPatterns(df, end_idx=last)", "bp = BreakoutPatterns(df)"),
    ("W08_frame_cut_at_moment", W, "bp = BreakoutPatterns(df, end_idx=last)",
     "bp = BreakoutPatterns(df, end_idx=last if known_at is None else min(last, int(known_at)))"),
    ("W09_detector_no_threshold", W, "ev = bp.detect_best_for_anchor(a, sd, thr)",
     "ev = bp.detect_best_for_anchor(a, sd, None)"),
    ("W10_start_ignored", W, "for a in range(max(0, int(start_idx)), last + 1):", "for a in range(0, last + 1):"),
    ("W11_no_early_stop", W, "        if best is not None and a >= best[1]:\n            break",
     "        if False:\n            break"),
    ("W12_end_ignored", W, "last = len(df) - 1 if end_idx is None else min(len(df) - 1, int(end_idx))",
     "last = len(df) - 1"),
    # --- the condition + the selection
    ("C01_price_side_off", PZ,
     "    if not passes_price_constraint(int(candle_idx), df, int(struct_direction), float(ref.price)):\n"
     "        return IC_REF_PAST_REFERENCE",
     "    if False:\n        return IC_REF_PAST_REFERENCE"),
    ("C02_bound_not_strict", PZ,
     "if ref.break_pattern_anchor_idx is not None and int(candle_idx) >= int(ref.break_pattern_anchor_idx):",
     "if ref.break_pattern_anchor_idx is not None and int(candle_idx) > int(ref.break_pattern_anchor_idx):"),
    ("C03_bound_off", PZ,
     "if ref.break_pattern_anchor_idx is not None and int(candle_idx) >= int(ref.break_pattern_anchor_idx):",
     "if False:"),
    ("C04_bound_reported_first", PZ,
     "    if not passes_price_constraint(int(candle_idx), df, int(struct_direction), float(ref.price)):\n"
     "        return IC_REF_PAST_REFERENCE\n"
     "    if ref.break_pattern_anchor_idx is not None and int(candle_idx) >= int(ref.break_pattern_anchor_idx):\n"
     "        return IC_REF_NOT_BEFORE_BREAK\n",
     "    if ref.break_pattern_anchor_idx is not None and int(candle_idx) >= int(ref.break_pattern_anchor_idx):\n"
     "        return IC_REF_NOT_BEFORE_BREAK\n"
     "    if not passes_price_constraint(int(candle_idx), df, int(struct_direction), float(ref.price)):\n"
     "        return IC_REF_PAST_REFERENCE\n"),
    ("C05_price_buy_inclusive", PZ, "return candle_high < reference_price", "return candle_high <= reference_price"),
    ("C06_price_sell_inclusive", PZ, "return candle_low > reference_price", "return candle_low >= reference_price"),
    ("C07_price_reads_wrong_side", PZ, "return candle_high < reference_price", "return candle_low < reference_price"),
    ("S01_reasons_not_computed", PZ,
     "ref_reasons = {i: ic_reference_reason(df, i, ic_reference, sd) for i in candidates}",
     "ref_reasons = {i: None for i in candidates}"),
    ("S02_switch_dead", PZ,
     "eligible = {i: touches[i] and (ref_reasons[i] is None or not _IC_REFERENCE_FILTER) for i in candidates}",
     "eligible = {i: touches[i] for i in candidates}"),
    ("S03_switch_on", PZ, "_IC_REFERENCE_FILTER = False", "_IC_REFERENCE_FILTER = True"),
    ("S04_reason_not_carried", PZ,
     "IPCandidate(ip=r, touches_zone=touches[r.ic_idx], ic_ref_reason=ref_reasons[r.ic_idx])",
     "IPCandidate(ip=r, touches_zone=touches[r.ic_idx])"),
    ("S05_pick_ignores_switch", PZ,
     "patterns = [c.ip for c in cands if eligible[c.ip.ic_idx] and c.ip.rank is not None]",
     "patterns = [c.ip for c in cands if c.touches_zone and c.ip.rank is not None]"),
    ("A3_drawn_no_reference", PZ,
     "ic_reference=fib_tracker.ic_reference(fib_state, df, evaluated_at=None, end_idx=lifecycle_cap),",
     "ic_reference=ICReference.NONE,"),
    ("A3_drawn_no_cap", PZ,
     "ic_reference=fib_tracker.ic_reference(fib_state, df, evaluated_at=None, end_idx=lifecycle_cap),",
     "ic_reference=fib_tracker.ic_reference(fib_state, df, evaluated_at=None, end_idx=None),"),
    ("A3_poi_fallback_no_cap", PZ,
     "        ip_records = select_ip_records(df, structure_events, fib_tracker, config, lifecycle_cap=lifecycle_cap)\n"
     "\n    # One record = one fib record",
     "        ip_records = select_ip_records(df, structure_events, fib_tracker, config)\n"
     "\n    # One record = one fib record"),
    ("A5_ipc_fallback_no_cap", PZ,
     "        ip_records = select_ip_records(df, structure_events, fib_tracker, config, lifecycle_cap=lifecycle_cap)\n"
     "    out: List[IPCandidateRecord] = []",
     "        ip_records = select_ip_records(df, structure_events, fib_tracker, config)\n"
     "    out: List[IPCandidateRecord] = []"),
    ("A5_reason_not_exported", PZ, "                ic_ref_reason=c.ic_ref_reason,\n", ""),
    ("A4_ms_no_reference", PZ, "ref = tracker.ic_reference(fib_state, df, evaluated_at=int(evaluated_at))",
     "ref = ICReference.NONE"),
    ("A4_ms_retrospective", PZ, "ref = tracker.ic_reference(fib_state, df, evaluated_at=int(evaluated_at))",
     "ref = tracker.ic_reference(fib_state, df, evaluated_at=None)"),
    ("A4_C01_outside_try", PZ,
     "    try:\n        df.attrs[\"timeframe\"]\n        if fib_state is None:\n            return None\n"
     "        ref = tracker.ic_reference(fib_state, df, evaluated_at=int(evaluated_at))\n",
     "    ref = None if fib_state is None else tracker.ic_reference(fib_state, df, evaluated_at=int(evaluated_at))\n"
     "    try:\n        df.attrs[\"timeframe\"]\n        if fib_state is None:\n            return None\n"),
    # --- the reference
    ("F01_cross_start_ignored", FT,
     "start_cycle = int(fib_state.meta[\"cross_start_cycle\"]) if fib_state.meta.get(\"cross_cycle\") else target",
     "start_cycle = target"),
    ("F02_cts_ref_from_cycle_2", FT, "        if start_cycle >= 1:\n            cts = self._cts_by_cycle",
     "        if start_cycle >= 2:\n            cts = self._cts_by_cycle"),
    ("F04_bound_of_target", FT, "anchor = self._est_pattern_anchor[(sid, start_cycle)]",
     "anchor = self._est_pattern_anchor.get((sid, target), self._est_pattern_anchor[(sid, start_cycle)])"),
    ("F05_known_strict", FT, "brk = (anchor, known) if (evaluated_at is None or known <= int(evaluated_at)) else None",
     "brk = (anchor, known) if (evaluated_at is None or known < int(evaluated_at)) else None"),
    ("F06_known_ignored", FT,
     "brk = (anchor, known) if (evaluated_at is None or known <= int(evaluated_at)) else None",
     "brk = (anchor, known)"),
    ("F07_limit_of_start_cycle", FT, "lim = self._structure_reference_limit(sid, target)",
     "lim = self._structure_reference_limit(sid, start_cycle)"),
    ("F08_t_below_m", FT, "return lim if (m is None or int(target_cycle) <= m) else None",
     "return lim if (m is None or int(target_cycle) < m) else None"),
    ("F09_m_ignored", FT, "return lim if (m is None or int(target_cycle) <= m) else None", "return lim"),
    ("F10_cross_allowed_used", FT, "        lim = self._structure_reference_limit(sid, target)\n",
     "        lim = self._limits[sid] if self._cross_allowed(sid, target) or target == 0 else None\n"
     "        lim = None if lim is None or lim.price is None else lim\n"),
    ("F11_reversal_bound_as_of", FT,
     "brk = (int(lim.break_pattern_anchor_idx), int(lim.break_confirmed_idx))",
     "brk = ((int(lim.break_pattern_anchor_idx), int(lim.break_confirmed_idx)) if evaluated_at is None or "
     "int(lim.break_confirmed_idx) <= int(evaluated_at) else None)"),
    ("F12_watch_from_moment", FT, "start = int(self._bos_by_cycle[(sid, 0)][0])",
     "start = int(self._bos_moment_by_cycle[(sid, 0)])"),
    ("F12b_watch_from_zero", FT, "start = int(self._bos_by_cycle[(sid, 0)][0])", "start = 0"),
    ("F13_watch_on_anchor", FT,
     "brk = find_cross_limit_break(df, float(lim.price), sd, start, end_idx=end_idx, known_at=evaluated_at)",
     "brk = find_cross_limit_break(df, float(lim.ref_anchor_price), sd, start, end_idx=end_idx, known_at=evaluated_at)"),
    ("F14_watch_not_as_of", FT,
     "brk = find_cross_limit_break(df, float(lim.price), sd, start, end_idx=end_idx, known_at=evaluated_at)",
     "brk = find_cross_limit_break(df, float(lim.price), sd, start, end_idx=end_idx, known_at=None)"),
    ("F15_watch_uncapped", FT,
     "brk = find_cross_limit_break(df, float(lim.price), sd, start, end_idx=end_idx, known_at=evaluated_at)",
     "brk = find_cross_limit_break(df, float(lim.price), sd, start, end_idx=None, known_at=evaluated_at)"),
    ("F16_tie_to_cts", FT, "chosen = min(refs, key=lambda r: (r[1], 0 if r[0] else 1))",
     "chosen = min(refs, key=lambda r: (r[1], 1 if r[0] else 0))"),
    ("F17_buy_takes_higher", FT, "chosen = min(refs, key=lambda r: (r[1], 0 if r[0] else 1))",
     "chosen = max(refs, key=lambda r: (r[1], 0 if r[0] else 1))"),
    ("F18_sell_takes_lower", FT, "chosen = min(refs, key=lambda r: (-r[1], 0 if r[0] else 1))",
     "chosen = min(refs, key=lambda r: (r[1], 0 if r[0] else 1))"),
    ("F19_rev_ref_is_p", FT, "refs.append((True, float(lim.ref_anchor_price), \"prev_structure_bos\", brk))",
     "refs.append((True, float(lim.price), \"prev_structure_bos\", brk))"),
    ("F20_xlim_ref_is_p", FT, "refs.append((True, float(lim.ref_anchor_price), \"cross_limit_anchor\", brk))",
     "refs.append((True, float(lim.price), \"cross_limit_anchor\", brk))"),
    ("F21_est_anchor_is_cts", FT, "self._est_pattern_anchor[(sid, cycle_id)] = ef.pattern_anchor_idx(event)",
     "self._est_pattern_anchor[(sid, cycle_id)] = int(cts_idx)"),
    ("F22_stamp_uncapped", FT, "ref = self.ic_reference(fib, df, evaluated_at=None, end_idx=lifecycle_cap)",
     "ref = self.ic_reference(fib, df, evaluated_at=None, end_idx=None)"),
    ("F23_stamp_keeps_stale", FT,
     "head = {k: v for k, v in fib.meta.items() if k not in ic_keys and k not in tail_keys}",
     "head = {k: v for k, v in fib.meta.items() if k not in tail_keys}"),
    ("F24_stamp_after_limit", FT, "                    **head,\n                    \"ic_ref_price\": ref.price,",
     "                    **head,\n                    **tail,\n                    \"ic_ref_price\": ref.price,"),
    # --- the limit + its builders
    ("L01_localized_adds", CCF, "break_pattern_anchor_idx=int(self.break_pattern_anchor_idx) - int(offset)",
     "break_pattern_anchor_idx=int(self.break_pattern_anchor_idx) + int(offset)"),
    ("L02_localized_conf_kept", CCF, "break_confirmed_idx=int(self.break_confirmed_idx) - int(offset))",
     "break_confirmed_idx=int(self.break_confirmed_idx))"),
    ("L03_no_anchor_assert", CCF, "        assert (self.ref_anchor_price is None) == (self.price is None), (",
     "        assert True, ("),
    ("L04_no_break_source_assert", CCF,
     "        assert has_break == (self.source == \"reversal_bos_frozen\"), (", "        assert True, ("),
    ("L05_no_ic_ref_break_assert", CCF,
     "        assert not has_break or self.price is not None, \"ICReference: a break without a reference price\"",
     "        pass"),
    ("M01_main_bos_after_reversal", SL,
     "             and ef.event_moment(ev) <= int(rev.idx)]", "             ]"),
    ("M02_main_first_bos", SL, "    last_bos = max(boses, key=ef.event_moment)\n    return CrossFibLimit(",
     "    last_bos = boses[0]\n    return CrossFibLimit("),
    ("M03_main_break_is_moment", SL,
     "break_pattern_anchor_idx=ef.pattern_anchor_idx(rev), break_confirmed_idx=int(rev.idx))",
     "break_pattern_anchor_idx=int(rev.idx), break_confirmed_idx=int(rev.idx))"),
    ("M04_main_bos_any_sid", SL,
     "if getattr(ev, \"type\", None) == \"BOS_CONFIRMED\" and int(ev.meta[\"structure_id\"]) == sid - 1",
     "if getattr(ev, \"type\", None) == \"BOS_CONFIRMED\""),
    ("X01_cts_anchor_is_threshold", CFL,
     "return CrossFibLimit(thr, \"prev_sub_cts_threshold\", from_pool_idx=from_pool, ref_anchor_price=conf_price)",
     "return CrossFibLimit(thr, \"prev_sub_cts_threshold\", from_pool_idx=from_pool, ref_anchor_price=thr)"),
    ("X03_bos_anchor_is_threshold", CFL, "        if e.type == \"BOS_CONFIRMED\":\n            bos_price",
     "        if e.type in (\"BOS_CONFIRMED\", \"BOS_THRESHOLD_UPDATED\"):\n            bos_price"),
    ("X04_rev_bos_after_reversal", CFL,
     "boses = [e for e in bounded.events if e.type == \"BOS_CONFIRMED\" and ef.event_moment(e) <= int(sc[0].idx)]",
     "boses = [e for e in bounded.events if e.type == \"BOS_CONFIRMED\"]"),
    ("X05_rev_break_local", CFL, "break_pattern_anchor_idx=ef.pattern_anchor_idx(sc[0]) + sb,",
     "break_pattern_anchor_idx=ef.pattern_anchor_idx(sc[0]),"),
    ("X06_rev_conf_local", CFL, "break_confirmed_idx=int(sc[0].idx) + sb)", "break_confirmed_idx=int(sc[0].idx))"),
    ("X07_parent_same_cycle", CFL, "            and int(e.meta[\"cycle_id\"]) == int(parent_cycle_id) - 1]",
     "            and int(e.meta[\"cycle_id\"]) <= int(parent_cycle_id) - 1][-1:]"),
    ("X08_tripwire_ignores_new", CFL,
     "        if ((got.price, got.source, got_from, got.ref_anchor_price, got.break_pattern_anchor_idx,\n"
     "             got.break_confirmed_idx)",
     "        if ((got.price, got.source, got_from, stored.ref_anchor_price, stored.break_pattern_anchor_idx,\n"
     "             stored.break_confirmed_idx)"),
    # --- coordinates
    ("E01_geometry_not_localized", EDM, "cross_fib_limit=cross_fib_limit.localized(slice_begin),",
     "cross_fib_limit=cross_fib_limit,"),
    ("E02_projection_not_localized", EDM, "cross_fib_limits={0: lim.localized(slice_begin)},",
     "cross_fib_limits={0: lim},"),
    ("E03_mirror_skips_break", EDM, "    \"ic_ref_break_pattern_anchor_idx\",\n    \"ic_ref_break_confirmed_idx\",\n",
     ""),
    # --- MS
    ("MS1_reversal_key_is_apply", MS, "not a robust source.\n                    \"pattern_anchor_idx\": int(ev.start_idx),",
     "not a robust source.\n                    \"pattern_anchor_idx\": int(apply_idx),"),
    ("MS2_pattern_refresh_moment", MS, "                self._refresh_poi_inners_for_cycle(int(apply_idx))",
     "                self._refresh_poi_inners_for_cycle(int(apply_idx) - 1)"),
    ("MS3_raw_refresh_moment", MS, "            self._refresh_poi_inners_for_cycle(int(i))",
     "            self._refresh_poi_inners_for_cycle(int(i) - 1)"),
    ("MS4_read_moment_is_anchor", MS, "tracker=self._fib_tracker, evaluated_at=int(moment))",
     "tracker=self._fib_tracker, evaluated_at=int(st.cts.idx))"),
    ("EF1_any_state_changed", EF, "if ev.type == \"STATE_CHANGED\" and ev.meta[\"to\"] == \"reversal\":",
     "if ev.type == \"STATE_CHANGED\":"),
    # --- the pipeline, the exports
    ("O1_no_stamp", ORC, "    fib_tracker.stamp_ic_references(df, lifecycle_cap)\n", ""),
    ("O2_stamp_uncapped", ORC, "    fib_tracker.stamp_ic_references(df, lifecycle_cap)\n",
     "    fib_tracker.stamp_ic_references(df)\n"),
    ("O3_select_uncapped", ORC,
     "ip_records = select_ip_records(df, events, fib_tracker, poi_config, lifecycle_cap=lifecycle_cap)",
     "ip_records = select_ip_records(df, events, fib_tracker, poi_config)"),
    ("H1_hover_on_unstamped", ZR, "    if \"cross_fib_limit_source\" not in m:\n        return None\n    if \"ic_ref",
     "    if \"ic_ref"),
    ("H2_hover_cycle_a", ZR, "cycle = f\", cycle {start_cycle - 1}\"", "cycle = f\", cycle {start_cycle}\""),
    ("H3_hover_cross_start_ignored", ZR,
     "start_cycle = int(m[\"cross_start_cycle\"]) if m.get(\"cross_cycle\") else int(fib_st.cycle_id)",
     "start_cycle = int(fib_st.cycle_id)"),
    ("H4_hover_break_text", ZR, "btxt = \"break: none\" if brk is None else f\"break {int(brk)}\"",
     "btxt = \"\" if brk is None else f\"break {int(brk)}\""),
    ("H5_hover_after_limit", ZR,
     "    ic_ref = fib_hover_ic_ref_text(fib_st)\n    if ic_ref is not None:\n        text = f\"{text}<br>{ic_ref}\"\n"
     "    limit = fib_hover_limit_text(fib_st)\n    return text if limit is None else f\"{text}<br>{limit}\"",
     "    ic_ref = fib_hover_ic_ref_text(fib_st)\n    limit = fib_hover_limit_text(fib_st)\n"
     "    text = text if limit is None else f\"{text}<br>{limit}\"\n"
     "    return text if ic_ref is None else f\"{text}<br>{ic_ref}\""),
    ("XZ1_csv_reason_empty", XZ, "            \"ic_ref_reason\": r.ic_ref_reason,", "            \"ic_ref_reason\": None,"),
]
