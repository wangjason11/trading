"""PLAN_W 9c question 2 — `w9c_measure.py` base vs `derived_ref`: per lens, every dumped part equal or not; the reads
whose returned zone changed; for a differing part, the entries removed / added (multiset of their JSON).

    PYTHONIOENCODING=utf-8 python engine_v2/plans/plan_w_inputs/w9c_derived_ref_diff.py BASE.json DERIVED_REF.json [OUT.txt]
"""
import collections
import json
import sys

base, var = (json.load(open(p, encoding="utf-8")) for p in sys.argv[1:3])
out = ["base vs derived_ref (a sibling read with a CTS_CONFIRMED winner returns the derived zone as of the read)"]
for r in var["refs"]:
    if r.get("returned") and (r["returned"]["outer"], r["returned"]["inner"]) != (r["ad_hoc"]["outer"], r["ad_hoc"]["inner"]):
        out.append(f"  changed read: {r['path']} read {r.get('lens_read')} parent {r.get('parent')} window {r.get('window')} "
                   f"dir {r['probe_direction']:+d}: ad hoc ({r['ad_hoc']['outer']}, {r['ad_hoc']['inner']}) -> "
                   f"derived ({r['returned']['outer']}, {r['returned']['inner']})")
for lens in ("H1.main >> M15.counter", "H1.main >> M15.confluence"):
    out.append(f"\n{lens}")
    for part in ("events", "kl_zones", "poi_zones", "ic_candidates", "sids", "wvmi", "wave_candles"):
        b, v = base[lens][part], var[lens][part]
        if b == v:
            out.append(f"   {part}: EQUAL ({len(b)})")
            continue
        cb = collections.Counter(json.dumps(x, sort_keys=True) for x in b)
        cv = collections.Counter(json.dumps(x, sort_keys=True) for x in v)
        out.append(f"   {part}: DIFFER ({len(b)} -> {len(v)}; removed {sum((cb - cv).values())}, added {sum((cv - cb).values())})")
        for s in list((cb - cv).elements())[:4]:
            out.append(f"      - {s[:300]}")
        for s in list((cv - cb).elements())[:4]:
            out.append(f"      + {s[:300]}")
text = "\n".join(out) + "\n"
print(text)
if len(sys.argv) > 3:
    open(sys.argv[3], "w", encoding="utf-8").write(text)
