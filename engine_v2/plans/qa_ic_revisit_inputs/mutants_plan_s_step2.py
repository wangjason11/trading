"""PLAN_S step 2 mutants (§3.3): each of the six implementations of "stroke 2 only on a candle after the arming one"
reverted to the same-candle form on its own — each must fail the suite.

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py engine_v2/plans/qa_ic_revisit_inputs/mutants_plan_s_step2.py
"""
_C = "  # stroke 2 — only on a candle AFTER the arming one"


def _loop(ind, cond1, cond2):
    """(old = the landed order, new = latch stroke 1 first → same-candle fill legal again)."""
    old = (f"{ind}if armed and {cond2}:{_C}\n{ind}    return True\n"
           f"{ind}if not armed and {cond1}:\n{ind}    armed = True\n")
    new = (f"{ind}if not armed and {cond1}:\n{ind}    armed = True\n"
           f"{ind}if armed and {cond2}:{_C}\n{ind}    return True\n")
    return old, new


T = "engine_v2/common/types.py"
P = "engine_v2/zones/poi_zones.py"
F, S = " " * 20, " " * 16
MUT = [
    ("s2_fast_bull", T, *_loop(F, "l_arr[pos] <= stroke1_level", "c_arr[pos] >= stroke2_level")),
    ("s2_fast_bear", T, *_loop(F, "h_arr[pos] >= stroke1_level", "c_arr[pos] <= stroke2_level")),
    ("s2_slow_bull", T, *_loop(S, 'float(df.loc[idx, "l"]) <= stroke1_level', 'float(df.loc[idx, "c"]) >= stroke2_level')),
    ("s2_slow_bear", T, *_loop(S, 'float(df.loc[idx, "h"]) >= stroke1_level', 'float(df.loc[idx, "c"]) <= stroke2_level')),
    ("s2_cache_bull", P,
     "                slice_c = c_arr[armed_pos + 1:]\n                stroke2_hits = np.where(slice_c >= stroke2_level)[0]\n"
     "                if stroke2_hits.size:\n                    confirmed_pos = armed_pos + 1 + int(stroke2_hits[0])\n",
     "                slice_c = c_arr[armed_pos:]\n                stroke2_hits = np.where(slice_c >= stroke2_level)[0]\n"
     "                if stroke2_hits.size:\n                    confirmed_pos = armed_pos + int(stroke2_hits[0])\n"),
    ("s2_cache_bear", P,
     "                slice_c = c_arr[armed_pos + 1:]\n                stroke2_hits = np.where(slice_c <= stroke2_level)[0]\n"
     "                if stroke2_hits.size:\n                    confirmed_pos = armed_pos + 1 + int(stroke2_hits[0])\n",
     "                slice_c = c_arr[armed_pos:]\n                stroke2_hits = np.where(slice_c <= stroke2_level)[0]\n"
     "                if stroke2_hits.size:\n                    confirmed_pos = armed_pos + int(stroke2_hits[0])\n"),
    # the cache's slice moved but its position offset not (or vice versa) — a half-applied edit
    ("s2_cache_bull_offset_only", P,
     "                    confirmed_pos = armed_pos + 1 + int(stroke2_hits[0])\n                    confirmed_fill_idx = _idx_from_pos(confirmed_pos)\n"
     "        elif inst.direction == -1:\n",
     "                    confirmed_pos = armed_pos + int(stroke2_hits[0])\n                    confirmed_fill_idx = _idx_from_pos(confirmed_pos)\n"
     "        elif inst.direction == -1:\n"),
]
# "On edge is ok" (the user, round 3): a close exactly AT the outer edge is stroke 2 — each comparison made strict.
MUT += [
    ("s2_edge_fast_bull", T, "if armed and c_arr[pos] >= stroke2_level:", "if armed and c_arr[pos] > stroke2_level:"),
    ("s2_edge_fast_bear", T, "if armed and c_arr[pos] <= stroke2_level:", "if armed and c_arr[pos] < stroke2_level:"),
    ("s2_edge_slow_bull", T, 'if armed and float(df.loc[idx, "c"]) >= stroke2_level:',
     'if armed and float(df.loc[idx, "c"]) > stroke2_level:'),
    ("s2_edge_slow_bear", T, 'if armed and float(df.loc[idx, "c"]) <= stroke2_level:',
     'if armed and float(df.loc[idx, "c"]) < stroke2_level:'),
    ("s2_edge_cache_bull", P, "np.where(slice_c >= stroke2_level)", "np.where(slice_c > stroke2_level)"),
    ("s2_edge_cache_bear", P, "np.where(slice_c <= stroke2_level)", "np.where(slice_c < stroke2_level)"),
]
# The landing review's surviving shapes (finding 2): stroke 1 must STAY latched across a candle that closes back inside
# the gap — "armed" re-evaluated per candle (only the candle right after the arming one could confirm), on each path.
MUT += [
    ("s2_latch_fast_bull", T, f"{F}if not armed and l_arr[pos] <= stroke1_level:\n{F}    armed = True\n",
     f"{F}armed = l_arr[pos] <= stroke1_level\n"),
    ("s2_latch_fast_bear", T, f"{F}if not armed and h_arr[pos] >= stroke1_level:\n{F}    armed = True\n",
     f"{F}armed = h_arr[pos] >= stroke1_level\n"),
    ("s2_latch_slow_bull", T, f'{S}if not armed and float(df.loc[idx, "l"]) <= stroke1_level:\n{S}    armed = True\n',
     f'{S}armed = float(df.loc[idx, "l"]) <= stroke1_level\n'),
    ("s2_latch_slow_bear", T, f'{S}if not armed and float(df.loc[idx, "h"]) >= stroke1_level:\n{S}    armed = True\n',
     f'{S}armed = float(df.loc[idx, "h"]) >= stroke1_level\n'),
    ("s2_latch_cache_bull", P, "                slice_c = c_arr[armed_pos + 1:]\n                stroke2_hits = np.where(slice_c >= ",
     "                slice_c = c_arr[armed_pos + 1:armed_pos + 2]\n                stroke2_hits = np.where(slice_c >= "),
    ("s2_latch_cache_bear", P, "                slice_c = c_arr[armed_pos + 1:]\n                stroke2_hits = np.where(slice_c <= ",
     "                slice_c = c_arr[armed_pos + 1:armed_pos + 2]\n                stroke2_hits = np.where(slice_c <= "),
]
