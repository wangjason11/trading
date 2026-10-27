"""Mutants for "WVMI back on the chart" step 4 (2026-10-26, `user_spec_20261024_wvmi_on_chart.md`): the per-LP rows —
`zones/wvmi.WVMITracker._set_lp_rows` + its three call sites, the orchestrator's end-reason wiring, the sub mirror's
shift of the row indices, and `debug/export_wvmi.export_wvmi_lps` (the collapsed filter, `relative_dir` at the LP
candle, the column order). Run from the repo root:

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \
        engine_v2/plans/qa_chart_edits_inputs/mutants_wvmi_step4.py

`old` / `new` use LF (the runner reads in text mode). Every mutant must be KILLED by the full suite. Left out as
EQUIVALENT: a temporary row's status candle = the next LP's `lp_idx` instead of its `as_of_idx` (only the FIRST LP can
be as of a later candle, and the first LP is never a "next" one); the H1 row's `relative_dir` None vs "" (both an
empty CSV cell); the lock's status candle = the cycle's lifecycle end instead of the BOS_{n+1} moment (equal on every
non-collapsed record: cycle n's end is cycle n+1's clamped start, and a floor above that moment collapses n too).
"""
WVMI = "engine_v2/zones/wvmi.py"
ORCH = "engine_v2/pipeline/orchestrator.py"
EDM = "engine_v2/multitf/entity_df_mutation.py"
EXP = "engine_v2/debug/export_wvmi.py"

_TEMPS = ("        rows = [row(as_of, lp, \"temporary\", \"replaced by a further close\", lp_seq[k + 1][0])\n")
_SAME_LP = "            if last is not None and lock_lp in (None, last[1]):\n"
_LOCKED_SAME = "                rows.append(row(*last, \"locked\", locked_reason, lock_idx))\n"
_UNLOCKED = ("                rows.append(row(*last, \"unlocked\", _UNLOCKED_REASON.get(reason, f\"cycle ended ({reason})\"), "
             "int(end)))\n")

MUT = [
    # the temporary rows
    ("temp_status_idx_own_as_of", WVMI, _TEMPS,
     "        rows = [row(as_of, lp, \"temporary\", \"replaced by a further close\", as_of)\n"),
    ("as_of_lp_swapped", WVMI, _TEMPS,
     "        rows = [row(lp, as_of, \"temporary\", \"replaced by a further close\", lp_seq[k + 1][0])\n"),
    ("temp_reason_old_wording", WVMI, _TEMPS,
     "        rows = [row(as_of, lp, \"temporary\", \"replaced by a deeper close\", lp_seq[k + 1][0])\n"),
    # the lock
    ("same_lp_swaps", WVMI, _SAME_LP, "            if last is not None and lock_lp is None:\n"),
    ("no_lock_lp_swaps", WVMI, _SAME_LP, "            if last is not None and lock_lp == last[1]:\n"),
    ("locked_status_idx_own_as_of", WVMI, _LOCKED_SAME,
     "                rows.append(row(*last, \"locked\", locked_reason, last[0]))\n"),
    ("swap_drops_the_temporary", WVMI,
     "                    rows.append(row(*last, \"temporary\", \"replaced by the locked LP\", lock_idx))\n",
     "                    pass\n"),
    ("swapped_lp_as_of_itself", WVMI,
     "                    rows.append(row(lock_idx, lock_lp, \"locked\", locked_reason, lock_idx))\n",
     "                    rows.append(row(lock_lp, lock_lp, \"locked\", locked_reason, lock_idx))\n"),
    ("locked_reason_this_cycle", WVMI,
     "            locked_reason = f\"cycle {rec.locked_by_cycle_id}'s BOS confirmed\"\n",
     "            locked_reason = f\"cycle {rec.bos_cycle_id}'s BOS confirmed\"\n"),
    # open / unlocked
    ("open_gets_a_status_idx", WVMI,
     "                rows.append(row(*last, \"open\", \"still moving at the data end\", None))\n",
     "                rows.append(row(*last, \"open\", \"still moving at the data end\", len(df) - 1))\n"),
    ("unlocked_at_end_minus_1", WVMI, _UNLOCKED, _UNLOCKED.replace("int(end)))", "int(end) - 1))")),
    ("unlocked_reason_raw", WVMI, _UNLOCKED,
     "                rows.append(row(*last, \"unlocked\", f\"cycle ended ({reason})\", int(end)))\n"),
    ("replacement_says_ended", WVMI, "    \"same_dir_replacement\": \"sub replaced\",\n",
     "    \"same_dir_replacement\": \"sub ended\",\n"),
    ("parent_end_says_replaced", WVMI, "    \"parent_end\": \"sub ended\",\n", "    \"parent_end\": \"sub replaced\",\n"),
    ("pm_without_weight", WVMI,
     "    return (float(df.loc[lp_idx, \"volume\"]) * _compute_last_wave_weight(df, lp_idx, pb_wave_dir)) / fp_volume\n",
     "    return float(df.loc[lp_idx, \"volume\"]) / fp_volume\n"),
    # the call sites
    ("create_no_rows", WVMI, "        self._set_lp_rows(record, key, df, lp_seq)\n", "        pass\n"),
    ("rescan_no_rows", WVMI, "            self._set_lp_rows(rec, key, df, lp_seq)\n", "            pass\n"),
    ("rescan_seq_not_stored", WVMI,
     "            new_lp_idx = lp_seq[-1][1] if lp_seq else None\n            self._lp_seq_by_key[key] = lp_seq\n",
     "            new_lp_idx = lp_seq[-1][1] if lp_seq else None\n"),
    ("lock_no_rows", WVMI,
     "        self._set_lp_rows(rec, key, df, self._lp_seq_by_key[key], lock=(ef.event_moment(bos_event), swapped_lp))\n",
     "        pass\n"),
    ("lock_lp_unread", WVMI,
     "        swapped_lp = lp_idx if lp_idx is not None and lp_idx in df.index else None\n",
     "        swapped_lp = None\n"),
    ("end_reason_not_passed", ORCH,
     "                          cycle_end_reason_by_key=cycle_end_reason_by_key)\n",
     "                          cycle_end_reason_by_key=None)\n"),
    # the sub mirror
    ("mirror_lp_unshifted", EDM, "            lp_idx=r.lp_idx + slice_begin,\n", "            lp_idx=r.lp_idx,\n"),
    ("mirror_as_of_unshifted", EDM, "            as_of_idx=r.as_of_idx + slice_begin,\n",
     "            as_of_idx=r.as_of_idx,\n"),
    ("mirror_status_idx_unshifted", EDM,
     "            lp_status_idx=r.lp_status_idx + slice_begin if r.lp_status_idx is not None else None,\n",
     "            lp_status_idx=r.lp_status_idx,\n"),
    # the CSV
    ("collapsed_rows_kept", EXP, "        if r.cycle_collapsed:\n            continue\n", ""),
    ("relative_dir_at_as_of", EXP,
     "                    sub_relative_dir_at(lp.lp_idx, sid_rec.relative_dir_segments, sid_rec.start_idx, int(sub_id))\n",
     "                    sub_relative_dir_at(lp.as_of_idx, sid_rec.relative_dir_segments, sid_rec.start_idx, int(sub_id))\n"),
    ("missing_sid_record_silent", EXP,
     "        sid_rec = sid_by_sub_id[int(sub_id)] if sub_id is not None else None\n",
     "        sid_rec = sid_by_sub_id.get(int(sub_id)) if sub_id is not None else None\n"),
    ("csv_pm_from_the_record", EXP, "                \"pullback_momentum\": lp.pullback_momentum,\n",
     "                \"pullback_momentum\": r.pullback_momentum,\n"),
    ("column_order", EXP, "    \"lp_idx\",\n    \"as_of_idx\",\n", "    \"as_of_idx\",\n    \"lp_idx\",\n"),
]
