"""Coordinate-hygiene census over a save / artifacts folder (M15 lens CSVs).

usage (repo root): python meta_census.py FOLDER [--all]
(Post-E·2, 2026-09-26: the coordinate-hygiene census; since Post-E·2 every M15 index key is
SHIFTED, so a non-empty UNSHIFTED table = a new slice-local key.)
For every M15 lens CSV with a `meta` column: every meta key whose value is an
int (or None) and whose name looks index-like (`*_idx`, `*_at`, `pb_start` — renamed
`last_pullback_apply_idx` in Post-E·4; kept for saves before it) is
classified SHIFTED (in the mirror's shift lists) or not, with its non-null cell
count per lens. For the family keys, each value is compared with the row's
sub `slice_begin` (= the sub's starting_idx - 50, from the subs CSV) to show
slice-local vs entity-absolute.
"""
import ast
import glob
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4)))  # repo root
from engine_v2.multitf import entity_df_mutation as edm  # noqa: E402

folder = sys.argv[1]
# + the pre-rename names the mirror shifted until Post-E·4 / Post-E·5 (2026-09-29d / 30), so a save before a rename
# does not show them as UNSHIFTED (a false "new slice-local key").
SHIFT_EV = set(edm._EVENT_META_IDX_KEYS) | {"pb_start", "cts_idx"}
SHIFT_Z = set(edm._ZONE_META_IDX_KEYS) | {"bos_idx", "cts_idx", "expanded_last_idx", "pb_reconfirm_idx"}
SHIFT_FIB = set(getattr(edm, "_FIB_META_IDX_KEYS", ("deactivated_at",))) | {"cycle1_bos_idx"}  # Post-E·2 list


def _idxlike(k):
    return k.endswith("_idx") or k.endswith("_at") or k == "pb_start"


def load(pattern):
    fs = glob.glob(os.path.join(folder, pattern))
    assert len(fs) == 1, (pattern, fs)
    return pd.read_csv(fs[0])


out = {}
for lens in ("confluence", "counter"):
    subs = load(f"*_M15_{lens}_subs.csv")
    sb = {int(r.sub_id): int(r.starting_idx) - 50 for r in subs.itertuples()}
    for kind, shift in (("structure_events", SHIFT_EV), ("kl_zones", SHIFT_Z),
                        ("poi_zones", SHIFT_Z), ("fib_lifecycle", SHIFT_FIB)):
        df = load(f"*_M15_{lens}_{kind}.csv")
        cnt = Counter()
        samples = defaultdict(list)
        for i, r in df.iterrows():
            m = ast.literal_eval(r["meta"]) if isinstance(r["meta"], str) else {}
            et = r["type"] if kind == "structure_events" else kind
            for k, v in m.items():
                if not _idxlike(k):
                    continue
                if v is None or (isinstance(v, float) and v != v):
                    continue
                if not isinstance(v, int):
                    cnt[(et, k, "NONINT:" + type(v).__name__)] += 1
                    continue
                tag = "SHIFTED" if k in shift else "UNSHIFTED"
                cnt[(et, k, tag)] += 1
                if tag == "UNSHIFTED" and len(samples[(et, k)]) < 3:
                    ref = r["idx"] if kind == "structure_events" else (
                        r.get("ic_idx") if kind == "poi_zones" else r.get("start_idx"))
                    samples[(et, k)].append((m.get("sub_id"), sb.get(m.get("sub_id")), ref, v))
        for key, n in sorted(cnt.items()):
            out[(lens, kind) + key] = n
        for key, s in samples.items():
            out[(lens, kind, "SAMPLE") + key] = s

rows = defaultdict(dict)
for key, n in out.items():
    lens, kind = key[0], key[1]
    if key[2] == "SAMPLE":
        continue
    rows[(kind,) + key[2:]][lens] = n
print(f"{'kind':16} {'type':24} {'key':22} {'tag':10} conf  ctr")
for key in sorted(rows):
    kind, et, k, tag = key
    if "--all" not in sys.argv and tag == "SHIFTED":
        continue
    print(f"{kind:16} {et:24} {k:22} {tag:10} {rows[key].get('confluence', 0):4} {rows[key].get('counter', 0):4}")
print("\nsamples (sub_id, slice_begin, row idx/ic_idx/start_idx, value):")
for key, s in sorted(out.items()):
    if key[2] == "SAMPLE" and key[0] == "confluence":
        print(" ", key[1], key[3], key[4], s)
