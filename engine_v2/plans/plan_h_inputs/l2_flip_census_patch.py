"""PLAN_L landing 2 — the flip census: the ranking rules' BEHAVIOUR (no format, no names) patched onto the L1 code,
to re-measure which existing pins L2 flips (PLAN_L §4.4 — the trial's 9 were measured at `92b8b61`, before L1).

SUPERSEDED since PLAN_P (2026-10-13): it patches code that no longer exists (the immediate-unfilled-imbalance
rule, then ranking rule 2). Kept as PLAN_L's record.

    python l2_flip_census_patch.py TREE      (TREE = a `git archive` of the L1 code; each edit must match exactly once)

Then `cd TREE && python -m pytest -q -p no:cacheprovider` (TREE holds a copy of oanda.cfg). The rules as decided
(PLAN_H §6 ICR-*): ranking rule 1 exclude pinbars (none left -> no IC); 2 the immediate unfilled imbalance (a filter);
3 maru > normal; 4 the most recent in play before the earliest unfilled sd imbalance after the OLDEST IN PLAY (ICR-8).
A ranked POI: no pattern floor, the pattern meta None-safe; the POI hover a stub line.
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


CENSUS_FN = '''def _census_ranking(df, fib_state, config, ic_candidates):
    """CENSUS: Step 3 (the ranking rules) over the IC candidates' checks when no pattern exists."""
    by_idx = {c.ic_idx: c for c in ic_candidates}
    if not by_idx:
        return None
    in_play = sorted(i for i in by_idx if str(df.loc[i, "candle_type"]) != "pinbar")
    if not in_play:
        return None
    if len(in_play) > 1:
        passing = filter_immediate_unfilled_imbalance(
            in_play, {i: by_idx[i].immediate_imbalance_first_c2_idx for i in in_play})
        if len(passing) == 1:
            in_play = passing
        else:
            in_play = prefer_candle_type(passing or in_play, df, include_pinbar=False)
            if len(in_play) > 1:
                in_play = [most_recent_before_imbalance(in_play, df, fib_state, config)[0]]
    return by_idx[in_play[0]].pattern


def select_ic(
'''

patch("engine_v2/zones/poi_zones.py", [
    ("def select_ic(\n", CENSUS_FN),
    ("""        df, fib_state, config, tuple(c for c in checks if c.ic_idx in ic_candidates))
    return ICSelection(checks, pick, tiebreak)
""", """        df, fib_state, config, tuple(c for c in checks if c.ic_idx in ic_candidates))
    if pick is None:
        pick = _census_ranking(df, fib_state, config, tuple(c for c in checks if c.ic_idx in ic_candidates))
    return ICSelection(checks, pick, tiebreak)
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
     '''    if meta["ip_rank"] is None:\n        return "No pattern · ranking (census)"\n'''
     '''    n_init, n_opp, n_cont = (int(x) for x in meta["ip_leg_candle_counts"].split("-"))\n'''),
])
print("[census patch] applied to", TREE)
