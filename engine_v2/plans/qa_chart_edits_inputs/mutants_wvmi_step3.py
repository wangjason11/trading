"""Mutants for "WVMI back on the chart" step 3 (2026-10-26, `user_spec_20261024_wvmi_on_chart.md`): the new LP rule —
`zones/wvmi._temporary_lp_sequence` (the furthest-close sequence with its first-LP routes), the lock warning, the
pullback CTS_CONFIRMED's `pattern_anchor_idx` (MS) and its accessor (`ef.pattern_anchor_idx`). Run from the repo root:

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \
        engine_v2/plans/qa_chart_edits_inputs/mutants_wvmi_step3.py

`old` / `new` use LF (the runner reads in text mode). Every mutant must be KILLED by the full suite. Left out as
EQUIVALENT: the updates starting after the first LP's own candle instead of its as-of candle — on the pattern route the
first LP is the furthest pattern candle, on the proximity "prior" route the furthest prior candle beyond p's close (so
p is not further either), so no candle in between can be further.
"""
WVMI = "engine_v2/zones/wvmi.py"
MS = "engine_v2/structure/market_structure.py"
EF = "engine_v2/structure/event_fields.py"

_FURTHER = "        return close(i) < ref_close if pb_wave_dir == -1 else close(i) > ref_close\n"
_CANDIDATE = "        return fp_idx + 1 <= i <= search_end and _is_qualified(df, i, pb_wave_dir)\n"
_PROX = (
    "        if prior is not None:\n"
    "            first = (p, prior)\n"
    "        elif candidate(p):\n"
    "            first = (p, p)\n"
)
_PATTERN = "        in_pattern = furthest(range(ef.pattern_anchor_idx(cts_event), p + 1))\n"
_FALLBACK = "        first = (p, in_pattern) if in_pattern is not None else first_after(p, close(fp_idx))\n"
_WARN = "            if lp_idx != rec.lp_idx:\n"

MUT = [
    # the furthest-close rule
    ("ties_move", WVMI, _FURTHER,
     "        return close(i) <= ref_close if pb_wave_dir == -1 else close(i) >= ref_close\n"),
    ("fp_allowed", WVMI, _CANDIDATE,
     "        return fp_idx <= i <= search_end and _is_qualified(df, i, pb_wave_dir)\n"),
    ("end_bound_plus_1", WVMI, _CANDIDATE,
     "        return fp_idx + 1 <= i <= search_end + 1 and _is_qualified(df, i, pb_wave_dir)\n"),
    ("create_unbounded", WVMI,
     "        lp_seq = _temporary_lp_sequence(df, fp_idx, pb_wave_dir, cts_event, self._lp_search_end((sid, cycle_id), df))\n",
     "        lp_seq = _temporary_lp_sequence(df, fp_idx, pb_wave_dir, cts_event, len(df) - 1)\n"),
    ("direction_only", WVMI, _CANDIDATE,
     "        return fp_idx + 1 <= i <= search_end and int(df.loc[i, \"direction\"]) == pb_wave_dir\n"),
    # the pattern route
    ("pattern_span_from_fp", WVMI, _PATTERN,
     "        in_pattern = furthest(range(fp_idx + 1, p + 1))\n"),
    ("pattern_last_candle", WVMI, _PATTERN,
     "        in_pattern = next((i for i in range(p, ef.pattern_anchor_idx(cts_event) - 1, -1) if candidate(i)), None)\n"),
    ("first_lp_as_of_itself", WVMI, _FALLBACK,
     "        first = (in_pattern, in_pattern) if in_pattern is not None else first_after(p, close(fp_idx))\n"),
    ("fallback_ref_confirmation_candle", WVMI, _FALLBACK,
     "        first = (p, in_pattern) if in_pattern is not None else first_after(p, close(p))\n"),
    ("no_fallback", WVMI, _FALLBACK,
     "        first = (p, in_pattern) if in_pattern is not None else None\n"),
    # the proximity route
    ("prox_prior_not_beyond_p", WVMI,
     "        prior = furthest(i for i in range(fp_idx + 1, p) if candidate(i) and further(i, close(p)))\n",
     "        prior = furthest(i for i in range(fp_idx + 1, p) if candidate(i))\n"),
    ("prox_self_before_prior", WVMI, _PROX,
     "        if candidate(p):\n"
     "            first = (p, p)\n"
     "        elif prior is not None:\n"
     "            first = (p, prior)\n"),
    ("prox_wait_ref_fp", WVMI,
     "            first = first_after(p, close(p))\n",
     "            first = first_after(p, close(fp_idx))\n"),
    # the lock warning
    ("warning_never", WVMI, _WARN, "            if False:\n"),
    ("warning_always", WVMI, _WARN, "            if True:\n"),
    # the event key and its accessor
    ("ms_anchor_is_the_moment", MS,
     "                    meta={\"via\": ev.name, \"pattern_anchor_idx\": int(ev.start_idx)},\n",
     "                    meta={\"via\": ev.name, \"pattern_anchor_idx\": int(apply_idx)},\n"),
    ("ms_anchor_is_the_pattern_end", MS,
     "                    meta={\"via\": ev.name, \"pattern_anchor_idx\": int(ev.start_idx)},\n",
     "                    meta={\"via\": ev.name, \"pattern_anchor_idx\": int(ev.end_idx)},\n"),
    ("ef_any_cts_confirmed", EF,
     "    if ev.type == \"CTS_CONFIRMED\" and ev.meta[\"confirmation_method\"] == \"pullback\":\n",
     "    if ev.type == \"CTS_CONFIRMED\":\n"),
]
