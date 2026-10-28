"""PLAN_W 9c — the report of `w9c_measure.py`'s two dumps (base, cts_export).

Question 1 (export / chart the sub CTS KL zones): per lens, the KL zones by kind in each mode, whether every base zone
is unchanged in the variant (the variant only ADDS), whether everything else (events, POIs, IC rows, subs, WVMI,
wave candles) is equal, and each added CTS zone (sub, cycle, bounds, lifecycle, base pattern).
Question 2 (the probe's reference zone): every sub probe reference read (base mode, deduplicated), its winning CTS
event, the ad-hoc zone used today, and for a CTS_CONFIRMED winner the derived zone as of the read / its INIT step /
at the sub's natural end; + the H1 main's reversal reads (derived today) vs ad hoc.

    PYTHONIOENCODING=utf-8 python engine_v2/plans/plan_w_inputs/w9c_diff.py BASE.json CTS_EXPORT.json [OUT.txt]
"""
import collections
import json
import sys

base, var = (json.load(open(p, encoding="utf-8")) for p in sys.argv[1:3])
out = ["== Question 1: the subs RETURN their CTS KL zones (`source_kinds=(\"BOS\", \"CTS\")`) — vs today"]
for lens in ("H1.main >> M15.counter", "H1.main >> M15.confluence"):
    b, v = base[lens], var[lens]
    kb = collections.Counter(z["kind"] for z in b["kl_zones"])
    kv = collections.Counter(z["kind"] for z in v["kl_zones"])
    sb = [json.dumps(z, sort_keys=True) for z in b["kl_zones"]]
    sv = collections.Counter(json.dumps(z, sort_keys=True) for z in v["kl_zones"])
    kept = all(sv[s] > 0 for s in sb)
    out.append(f"\n{lens}: KL zones {dict(kb)} -> {dict(kv)}; every base zone unchanged in the variant: {kept}")
    for part in ("events", "poi_zones", "ic_candidates", "sids", "wvmi", "wave_candles"):
        out.append(f"   {part}: {'EQUAL' if b[part] == v[part] else 'DIFFER'} ({len(b[part])} / {len(v[part])})")
    added = [z for z in v["kl_zones"] if json.dumps(z, sort_keys=True) not in set(sb)]
    seen = set()
    out.append(f"   added zones ({len(added)}; one row per (sub, cycle)):")
    for z in sorted(added, key=lambda z: (str(z["meta"].get("sub_id")), z["meta"].get("cycle_id"))):
        m = z["meta"]
        k = (m.get("sub_id"), m.get("cycle_id"))
        if k in seen:
            continue
        seen.add(k)
        steps = m.get("bounds_steps") or []
        out.append(f"     {z['kind']} sub {m.get('sub_id')} c{m.get('cycle_id')} {z['side']} [{z['bottom']:.5f}, {z['top']:.5f}] "
                   f"base {m.get('base_pattern')} anchor {m.get('anchor_idx')} confirmed {m.get('confirmed_idx')} "
                   f"end {m.get('end_idx')} ({m.get('end_reason')}) status {m.get('status')} steps {len(steps)}")

out.append("\n== Question 2: the sub probe's reference zone — ad hoc today vs the derived CTS KL zone")
refs = base["refs"]
seen, rows = set(), []
for r in refs:
    key = json.dumps({k: r.get(k) for k in ("path", "lens_read", "parent", "probe_direction", "window", "sub_pool_idx",
                                             "reversal_idx", "sid", "n_frame")}, sort_keys=True)
    if key in seen:
        continue
    seen.add(key)
    rows.append(r)
sub_rows = [r for r in rows if r["path"] in ("sibling", "reversal")]
main_rows = [r for r in rows if r["path"] == "main_reversal"]
out.append(f"{len(refs)} reads recorded, {len(sub_rows)} distinct sub reads, {len(main_rows)} distinct structure_engine "
           f"reversal reads; sub winners by type: {dict(collections.Counter(r.get('winner_type') for r in sub_rows))}")


def oi(z):
    return "-" if z is None else f"({z['outer']:.5f}, {z['inner']:.5f})"


n_conf = n_asof_eq = n_init_eq = n_end_eq = 0
for r in sub_rows:
    where = (f"{r['path']:<8} " + (f"read {r['lens_read'].split('.')[-1]:<10} parent {tuple(r['parent'])} window "
                                    f"{tuple(r['window'])}" if r["path"] == "sibling" else
                                    f"pool#{r['sub_pool_idx']} R={r['reversal_idx']}"))
    line = (f"  {where} dir {r['probe_direction']:+d} | winner {r.get('winner_type')} c{r.get('winner_cycle')} "
            f"moment {r.get('winner_moment')} anchor {r.get('winner_anchor')} | ad hoc {oi(r['ad_hoc'])}")
    d, e = r.get("derived_as_of"), r.get("derived_natural_end")
    if r.get("winner_type") == "CTS_CONFIRMED":
        n_conf += 1
        asof = None if d is None else d["now"]
        init = None if d is None else d.get("init")
        fin = None if e is None else e["now"]
        same = lambda z: z is not None and r["ad_hoc"] is not None and (z["outer"], z["inner"]) == (r["ad_hoc"]["outer"], r["ad_hoc"]["inner"])
        n_asof_eq += same(asof)
        n_init_eq += same(init)
        n_end_eq += same(fin)
        line += (f" | derived as of the read {oi(asof)}{' =' if same(asof) else ' DIFFERS'}"
                 f" (INIT {oi(init)}, {None if d is None else d['n_steps']} step(s), base {None if d is None else d['base_pattern']})"
                 f" | at the natural end {oi(fin)}{' =' if same(fin) else ' DIFFERS'} ({None if e is None else e['n_steps']} step(s))")
    out.append(line)
out.append(f"CTS_CONFIRMED winners: {n_conf}; the derived zone as of the read = ad hoc: {n_asof_eq}; its INIT step = ad hoc: "
           f"{n_init_eq}; the natural-end zone = ad hoc: {n_end_eq}")
out.append("\nThe structure_engine reversal reads (the H1 main — it already passes its derived CTS zones):")
for r in main_rows:
    g, a = r["got (derived when CONFIRMED)"], r["ad_hoc"]
    out.append(f"  sid {r['sid']} dir {r['probe_direction']:+d} frame n={r['n_frame']} CTS zones passed {r['n_cts_zones_passed']}: "
               f"got {oi(g)} source {None if g is None else g['source']} anchor {None if g is None else g['anchor_idx']} | "
               f"ad hoc {oi(a)}{' =' if g and a and (g['outer'], g['inner']) == (a['outer'], a['inner']) else ' DIFFERS'}")
text = "\n".join(out) + "\n"
print(text)
if len(sys.argv) > 3:
    open(sys.argv[3], "w", encoding="utf-8").write(text)
