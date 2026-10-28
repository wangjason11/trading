"""PLAN_W 9b — the concrete example for the user's decision: one sub cycle that ENDED before its CTS confirmed, on a
window that has one (2025-08-01 -> 2025-09-20: sub 2, cycle 1). Its events (M15, entity-absolute), the IC candle and
what the relaxed gate would add, from the two `w9b_variant_measure.py` dumps.

    PYTHONIOENCODING=utf-8 python engine_v2/plans/plan_w_inputs/w9b_example_table.py BASE.json VARIANT.json SUB CYCLE [OUT.txt]
"""
import json
import sys

base, var = (json.load(open(p, encoding="utf-8")) for p in sys.argv[1:3])
sub, cyc = int(sys.argv[3]), int(sys.argv[4])
ent = "H1.main >> M15.confluence"
b, v = base[ent], var[ent]
times = b["time"]
out = [f"window {base['window'][0]} -> {base['window'][1]}, {ent}: sub {sub} cycle {cyc}", "",
       "   M15 idx  time (UTC)        event                price     meta"]
for e in b["events"]:
    m = e["meta"]
    if m.get("sub_id") != sub or m.get("cycle_id") != cyc:
        continue
    if e["type"] not in ("BOS_CONFIRMED", "CTS_ESTABLISHED", "CTS_UPDATED", "CTS_CONFIRMED", "STATE_CHANGED",
                         "REVERSAL_CANDIDATE", "RANGE_STARTED"):
        continue
    keep = {k: m.get(k) for k in ("cts_anchor_idx", "bos_anchor_idx", "confirmed_at", "via", "to") if m.get(k) is not None}
    out.append(f"   {e['idx']:>7}  {times[e['idx']][:16]}  {e['type']:<20} {str(m.get('price', '')):<9} {keep}")
fib_b = [r for r in b["ic_candidates"] if (r.get("meta") or {}).get("sub_id") == sub and r["cycle_id"] == cyc]
fib_v = [r for r in v["ic_candidates"] if (r.get("meta") or {}).get("sub_id") == sub and r["cycle_id"] == cyc]
poi_b = [p for p in b["poi_zones"] if p["sub_id"] == sub and p["cycle_id"] == cyc]
poi_v = [p for p in v["poi_zones"] if p["sub_id"] == sub and p["cycle_id"] == cyc]
out += ["", f"today (base):        IC rows {len(fib_b)}, POI zones {len(poi_b)}",
        f"relaxed gate:        IC rows {len(fib_v)} ({sum(1 for r in fib_v if r.get('is_ic_candidate'))} IC candidates), "
        f"POI zones {len(poi_v)}"]
for p in poi_v:
    m = p["meta"]
    out.append(f"   POI: IC {p['ic_idx']} ({times[p['ic_idx']][:16]}), {p['side']} [{p['bottom']:.5f}, {p['top']:.5f}], "
               f"pattern {m.get('ip_rank')} ({m.get('ip_outcome')}), band {m.get('ic_zone_band')}")
    out.append(f"        active from {m.get('confirmed_idx')} ({times[m['confirmed_idx']][:16]}) to its cycle's end "
               f"{m.get('end_idx')} ({times[m['end_idx']][:16]}, {m.get('end_reason')}); "
               f"activation_history {[(h.get('idx'), h.get('active'), h.get('reason')) for h in m.get('activation_history') or []]}")
for r in fib_v:
    if r.get("picked"):
        out.append(f"   the picked row: IC {r['ic_idx']} candle_type={r.get('candle_type')} fib bos {r['bos_anchor_idx']} "
                   f"({times[r['bos_anchor_idx']][:16]}) -> cts {r['cts_anchor_idx']} ({times[r['cts_anchor_idx']][:16]})")
text = "\n".join(out) + "\n"
print(text)
if len(sys.argv) > 5:
    open(sys.argv[5], "w", encoding="utf-8").write(text)
