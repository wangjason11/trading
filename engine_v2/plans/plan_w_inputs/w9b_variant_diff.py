"""PLAN_W 9b — diff two `w9b_variant_measure.py` dumps (base vs gate_any) of one window.

Per entity (H1.main, M15.counter, M15.confluence): the POI zones and IC records added / removed / changed (keyed by
(sub_id, structure_id, cycle_id)), and whether the structure events, KL zones and subs are equal (they must be: the
gate is read only by `select_ic_records`, and the proximity overlay reads a sub's POI only as of its cycle's
CTS_CONFIRMED moment, when its fib is locked). For each new POI: its IC, band, lifecycle and activation with times.

    PYTHONIOENCODING=utf-8 python engine_v2/plans/plan_w_inputs/w9b_variant_diff.py BASE.json VARIANT.json [OUT.txt]
"""
import json
import sys
from collections import defaultdict

base, var = (json.load(open(p, encoding="utf-8")) for p in sys.argv[1:3])
out = []


def key(d):
    return (d.get("sub_id"), d.get("structure_id"), d.get("cycle_id"))


def t(times, i):
    try:
        return times[int(i)][:16]
    except (TypeError, ValueError, IndexError):
        return "-"


out.append(f"window {base['window'][0]} -> {base['window'][1]}: base vs gate_any")
for ent in ("H1.main", "H1.main >> M15.counter", "H1.main >> M15.confluence"):
    if ent not in base:
        continue
    b, v = base[ent], var[ent]
    times = b["time"]
    out.append(f"\n== {ent}")
    for part in ("events", "kl_zones", "sids"):
        out.append(f"   {part}: {'EQUAL' if b[part] == v[part] else 'DIFFER'} ({len(b[part])} / {len(v[part])})")
    bp = {key(p): p for p in b["poi_zones"]}
    vp = {key(p): p for p in v["poi_zones"]}
    out.append(f"   poi_zones: {len(bp)} -> {len(vp)}")
    for k in sorted(set(bp) | set(vp), key=str):
        if bp.get(k) == vp.get(k):
            continue
        if k not in vp:
            out.append(f"     REMOVED {k}")
            continue
        p = vp[k]
        m = p["meta"]
        hist = m.get("activation_history") or []
        acts = [(h.get("idx"), h.get("active"), h.get("reason")) for h in hist] if isinstance(hist, list) else hist
        out.append(f"     {'NEW' if k not in bp else 'CHANGED'} sub {k[0]} sid {k[1]} cyc {k[2]}: IC {p['ic_idx']} "
                   f"({t(times, p['ic_idx'])}) {p['side']} [{p['bottom']:.5f}, {p['top']:.5f}] band={m.get('ic_zone_band')} "
                   f"ip_outcome={m.get('ip_outcome')} ip_rank={m.get('ip_rank')} ranking_rule={m.get('ranking_rule')}")
        out.append(f"         cts_established {m.get('cts_established_idx')} ({t(times, m.get('cts_established_idx'))}), "
                   f"confirmed_idx {m.get('confirmed_idx')} ({t(times, m.get('confirmed_idx'))}), end {m.get('end_idx')} "
                   f"({t(times, m.get('end_idx'))}) end_reason={m.get('end_reason')} status={m.get('status')}")
        out.append(f"         activation_history: {acts}")
    bi, vi = defaultdict(list), defaultdict(list)
    for r in b["ic_candidates"]:
        bi[key(r) if r.get("sub_id") is not None else (r.get("meta", {}).get("sub_id"), r["structure_id"], r["cycle_id"])].append(r)
    for r in v["ic_candidates"]:
        vi[key(r) if r.get("sub_id") is not None else (r.get("meta", {}).get("sub_id"), r["structure_id"], r["cycle_id"])].append(r)
    out.append(f"   ic records: {len(bi)} -> {len(vi)} (rows {len(b['ic_candidates'])} -> {len(v['ic_candidates'])})")
    for k in sorted(set(bi) | set(vi), key=str):
        if bi.get(k) == vi.get(k):
            continue
        rows = vi.get(k, [])
        picked = [r for r in rows if r.get("picked")]
        noic = [r for r in rows if r.get("ic_idx") is None]
        out.append(f"     {'NEW' if k not in bi else 'CHANGED'} ic record sub {k[0]} sid {k[1]} cyc {k[2]}: {len(rows)} rows, "
                   f"candidates {sum(1 for r in rows if r.get('is_ic_candidate'))}, "
                   f"pick {[r['ic_idx'] for r in picked]}, no-IC reason {[r.get('reason') for r in noic]}, "
                   f"ip_outcome {sorted(set(str(r.get('ip_outcome')) for r in rows))}, "
                   f"fib bos {rows[0]['bos_anchor_idx'] if rows else '-'} -> cts {rows[0]['cts_anchor_idx'] if rows else '-'}")
text = "\n".join(out) + "\n"
print(text)
if len(sys.argv) > 3:
    open(sys.argv[3], "w", encoding="utf-8").write(text)
