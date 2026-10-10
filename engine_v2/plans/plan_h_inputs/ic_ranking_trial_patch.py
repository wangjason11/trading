"""The IC identification revamp — the TRIAL replay's behaviour patch (no restructure, no names): Step 3, the ranking
rules, applied in BOTH consumers of `select_ip_ic` (the drawn POI and MS's in-flight read), the user's rules as
decided (PLAN_H §6 ICR-*):
  ranking rule 1 exclude pinbars (none left -> no IC); 2 the immediate unfilled imbalance (a filter: one -> final,
  >=2 -> only they, none -> all); 3 maru > normal; 4 the most recent in play before the earliest unfilled sd imbalance
  after the OLDEST IN PLAY (ICR-8). `ranking_rule` None when Step 1 had ONE non-pinbar candidate (ICR-10).
A Step-3 POI has no pattern term in its activation floor (ICR-2): the floor gets `ic_idx` (neutral in the max).
The picked CSV row's `reason` = "RANKED:<rule>" (trial marker only).

    python ic_ranking_trial_patch.py TREE      (TREE = a `git archive` of 92b8b61; each edit must match exactly once)

Then (a SHORT %TEMP% dir — MAX_PATH): `cd WORK && python REPO/engine_v2/plans/plan_h_inputs/replay_variant_h.py TREE
REPO/engine_v2/zones/poi_zones.py log.json --nolog > run.log` (WORK holds a copy of oanda.cfg), then
`cmp_save.py <S_TBU> WORK/artifacts/debug WORK/artifacts/charts --names-from <S_TBU>`. Result: PLAN_H §8.9
"The IC identification revamp" (the trial replay).
"""
import sys
from pathlib import Path

TREE = Path(sys.argv[1])


def patch(rel, edits):
    p = TREE / rel
    src = p.read_text(encoding="utf-8")
    for old, new in edits:
        n = src.count(old)
        assert n == 1, (rel, n, old[:100])
        src = src.replace(old, new)
    p.write_text(src, encoding="utf-8")


TRIAL_FN = '''def _trial_ranking(df, fib_state, config, cands, eligible):
    """TRIAL: Step 3 (the ranking rules) over the IC candidates when no pattern exists."""
    import dataclasses
    sd = int(fib_state.struct_direction)
    cts = int(fib_state.cts_idx)
    by_idx = {c.ip.ic_idx: c for c in cands if eligible[c.ip.ic_idx]}
    if not by_idx:
        return IPSelection(cands, None, None)
    ctype = {i: str(df.loc[i, "candle_type"]) for i in by_idx}
    in_play = sorted(i for i in by_idx if ctype[i] != "pinbar")
    if not in_play:
        return IPSelection(cands, None, None)
    rule = "exclude_pinbar" if len(by_idx) > 1 else None
    if len(in_play) > 1:
        passing = [i for i in in_play if by_idx[i].immediate_imbalance_first_c2_idx is not None]
        if len(passing) == 1:
            in_play, rule = passing, "immediate_unfilled_imbalance"
        else:
            in_play = passing or in_play
            best = min(("maru", "normal").index(ctype[i]) for i in in_play)
            by_type = [i for i in in_play if ("maru", "normal").index(ctype[i]) == best]
            if len(by_type) == 1:
                in_play, rule = by_type, "ic_candle_type"
            else:
                oldest = min(by_type)
                ref = min(iter_unfilled_imbalances(df, oldest + 1, cts, cts, config.fill_threshold,
                                                   direction=sd, evaluated_at=None),
                          key=lambda inst: inst.start_idx)
                in_play = [max(i for i in by_type if i < ref.start_idx)]
                rule = "most_recent_before_imbalance"
    pick = dataclasses.replace(by_idx[in_play[0]].ip, reason=f"RANKED:{rule}")
    cands2 = tuple(dataclasses.replace(c, ip=pick) if c.ip.ic_idx == pick.ic_idx else c for c in cands)
    return IPSelection(cands2, pick, None)


def select_ip_ic(
'''

patch("engine_v2/zones/poi_zones.py", [
    ("def select_ip_ic(\n", TRIAL_FN),
    ("""    if not patterns:
        return IPSelection(cands, None, None)
""", """    if not patterns:
        return _trial_ranking(df, fib_state, config, cands, eligible)
"""),
    ("            pattern_confirmed_idx=int(pick.pattern_confirmed_idx),\n",
     "            pattern_confirmed_idx=(int(pick.pattern_confirmed_idx) if pick.pattern_confirmed_idx is not None\n"
     "                                   else ic_idx),\n"),
    ('                "pattern_anchor_idx": int(pick.pattern_anchor_idx),\n',
     '                "pattern_anchor_idx": None if pick.pattern_anchor_idx is None else int(pick.pattern_anchor_idx),\n'),
    ('                "opp_leg_first_idx": int(pick.opp_leg_first_idx),\n',
     '                "opp_leg_first_idx": None if pick.opp_leg_first_idx is None else int(pick.opp_leg_first_idx),\n'),
    ('                "pattern_end_idx": int(pick.pattern_end_idx),\n',
     '                "pattern_end_idx": None if pick.pattern_end_idx is None else int(pick.pattern_end_idx),\n'),
    ('                "pattern_confirmed_idx": int(pick.pattern_confirmed_idx),\n',
     '                "pattern_confirmed_idx": (None if pick.pattern_confirmed_idx is None\n'
     '                                          else int(pick.pattern_confirmed_idx)),\n'),
])

patch("engine_v2/charting/_zone_render.py", [
    ('''    n_init, n_opp, n_cont = (int(x) for x in meta["ip_leg_candle_counts"].split("-"))\n''',
     '''    if meta["ip_rank"] is None:\n        return f"No pattern · IC {int(ic_idx)} · ranking (trial)"\n'''
     '''    n_init, n_opp, n_cont = (int(x) for x in meta["ip_leg_candle_counts"].split("-"))\n'''),
])
print("[trial patch] applied to", TREE)
