"""Patch a `git archive` copy of the tree (argv[1]) so the imbalance fill's stroke 2 must come on a candle STRICTLY
AFTER the arming candle (the user's round-3 decision, 2026-10-21: "the 2nd condition for filled must be met by a
different candle after the armed candle") — a VARIANT replay to measure that change by itself. Run it AFTER
`make_variant_fill65.py` on the same copy (the order: 65 % lands first, this one on top).

Every implementation of the two-stroke rule (grep `gap_size *` 2026-10-21): `ImbalanceInstance.is_filled`'s positional
fast path + its label-based fallback (`common/types.py`, 4 loops: check stroke 2 BEFORE latching stroke 1 on the same
candle) and `_compute_fill_idx_cache` (`zones/poi_zones.py`, the POI activation sweep's numpy scan: stroke 2 from
armed_pos + 1). Each edit is exact-count.

    git archive HEAD | tar -x -C <short %TEMP% dir>; python make_variant_fill65.py <dir>; python <this> <dir>
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])


def patch(rel, pairs):
    p = ROOT / rel
    b = p.read_bytes().decode("utf-8")
    nl = "\r\n" if "\r\n" in b else "\n"  # the archive may carry CRLF (Windows checkout)
    for old, new, n in pairs:
        old, new = old.replace("\n", nl), new.replace("\n", nl)
        assert b.count(old) == n, (rel, old, b.count(old))
        b = b.replace(old, new)
    p.write_bytes(b.encode("utf-8"))


ind_fast = " " * 20
ind_slow = " " * 16
pairs = []
for ind, cond_l, cond_h, c_bull, c_bear in (
    (ind_fast, "l_arr[pos] <= stroke1_level", "h_arr[pos] >= stroke1_level",
     "c_arr[pos] >= stroke2_level", "c_arr[pos] <= stroke2_level"),
    (ind_slow, 'float(df.loc[idx, "l"]) <= stroke1_level', 'float(df.loc[idx, "h"]) >= stroke1_level',
     'float(df.loc[idx, "c"]) >= stroke2_level', 'float(df.loc[idx, "c"]) <= stroke2_level'),
):
    for cond1, cond2 in ((cond_l, c_bull), (cond_h, c_bear)):
        old = (f"{ind}if not armed and {cond1}:\n{ind}    armed = True\n"
               f"{ind}if armed and {cond2}:\n{ind}    return True\n")
        new = (f"{ind}if armed and {cond2}:\n{ind}    return True\n"
               f"{ind}if not armed and {cond1}:\n{ind}    armed = True\n")
        pairs.append((old, new, 1))
patch("engine_v2/common/types.py", pairs)
patch("engine_v2/zones/poi_zones.py", [
    ("slice_c = c_arr[armed_pos:]", "slice_c = c_arr[armed_pos + 1:]", 2),
    ("confirmed_pos = armed_pos + int(stroke2_hits[0])", "confirmed_pos = armed_pos + 1 + int(stroke2_hits[0])", 2),
])
print("patched: stroke 2 strictly after the arming candle (types.py 4 loops, poi_zones.py fill cache)")
