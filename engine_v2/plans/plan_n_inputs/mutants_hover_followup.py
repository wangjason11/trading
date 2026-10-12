"""The PLAN_N hover follow-up as landed (option B: the hover lines snapshot at each event's REPLAY candle — the hover
line = the box, candle for candle; `followup_hover_per_structure.md` "The landing decision") — its mutants, plus the
PLAN_N landing review's LR1-LR3 re-run on the landed code (LR1 re-anchored: the hover builder's old
`all_range_events = ...` line is gone). Full suite each; every one must be KILLED.

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py \
        engine_v2/plans/plan_n_inputs/mutants_hover_followup.py --jobs 4 --out <a short %TEMP% dir>

Not a mutant under B: the note's "H1" (restore the all-structures snapshot) — B's snapshot IS of every structure; with
every snapshot at its replay candle they are written in replay order, so it is correct (equivalent to per-structure).
"""
EP = "engine_v2/charting/export_plotly.py"

MUT = [
    # B itself: the snapshot back at the event's idx (= the pre-follow-up pass: both symptoms return)
    ("HB1_snapshot_at_the_event_idx", EP,
     "            snaps[replay_idx] = {s: dict(r) for s, r in open_ranges.items()}\n",
     "            snaps[idx] = {s: dict(r) for s, r in open_ranges.items()}\n"),
    # the forward-fill over visible candles only (a snapshot from before a slice's first candle dropped)
    ("HB2_forward_fill_visible_candles_only", EP,
     "        for idx in sorted(idx_set | set(snaps)):\n",
     "        for idx in sorted(idx_set):\n"),
    # no forward-fill (a candle without an event of its own shows nothing)
    ("HB3_no_forward_fill", EP,
     "                range_by_candle[idx] = cur\n",
     "                range_by_candle[idx] = snaps.get(idx, {})\n"),
    # the helper hands out the event's idx as its replay candle (the order unchanged)
    ("HB4_helper_replay_idx_is_the_event_idx", EP,
     "    return [(key[0], ev) for key, ev in keyed]\n",
     "    return [(int(ev.idx), ev) for key, ev in keyed]\n"),
    # a STARTED's last update = its moment, not its anchor (pre-existing behaviour, pinned by the confirm-candle test)
    ("HB5_started_last_update_at_its_moment", EP,
     '                    "last_update_idx": start_idx,\n',
     '                    "last_update_idx": idx,\n'),
    # LR1 re-anchored: the hover builder back on a plain idx sort
    ("LR1_hover_order_back_on_an_idx_sort", EP,
     "        for replay_idx, ev in _range_events_with_replay_idx(structure_events):\n",
     "        for replay_idx, ev in [(int(e.idx), e) for e in sorted(range_started + range_updated + range_reset,\n"
     "                               key=lambda e: (int(e.idx), _RANGE_TYPE_PRIORITY[e.type]))]:\n"),
    ("LR2_guard_not_per_structure", EP,
     '        sid = int(ev.meta.get("structure_id", 0))\n        idx = int(ev.idx)\n        if t == "RANGE_STARTED":',
     '        sid = 0\n        idx = int(ev.idx)\n        if t == "RANGE_STARTED":'),
    ("LR3_guard_takes_a_later_reset", EP,
     "            if r is not None and r <= idx:\n",
     "            if r is not None:\n"),
]
