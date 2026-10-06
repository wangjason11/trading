"""PLAN_H §8.4 — summarise a `replay_variant_h.py` resolver log (PR8 / PR12).

    python engine_v2/plans/plan_h_inputs/resolver_log_report.py LOG.json [--rows]

Per caller (main / sub_geometry / probe_phase2): calls, the variant's inners vs A's (equal / IP ⊂ A / IP ∉ A / IP
none while A had some / both empty), the call times (variant vs A, and `select_ip_ic` alone). Then PR12: per
(tf, frame start, sid, cycle) the as-of picks of the resolver against the FINAL drawn pick on the same frame start
(the `drawn` records — V1 / V3 only), with the anchors.
"""
import collections
import json
import statistics
import sys

L = json.load(open(sys.argv[1]))
ROWS = "--rows" in sys.argv
res = [r for r in L if r["kind"] == "resolver"]
drawn = [r for r in L if r["kind"] == "drawn"]
errs = [r for r in L if r["kind"] == "select_error"]
print(f"records: {len(L)}; resolver {len(res)}, drawn {len(drawn)}, csv {sum(r['kind'] == 'csv' for r in L)}, "
      f"select_error {len(errs)}")
for e in errs:
    print("  ERROR", e)


def cls(r):
    a, v = r["today"], r["inners"]
    if a == v:
        return "equal" if v else "both_empty"
    if not v:
        return "ip_none_A_some"
    if v[0] in a:
        return "ip_in_A"
    return "ip_not_in_A" if a else "ip_some_A_none"


by = collections.defaultdict(list)
for r in res:
    by[r["caller"]].append(r)
for caller, rs in sorted(by.items()):
    c = collections.Counter(cls(r) for r in rs)
    ms = [r["ms"] for r in rs]
    tms = [r["today_ms"] for r in rs]
    ipms = [r["select"]["ip_ms"] for r in rs if "select" in r]
    print(f"\n{caller}: {len(rs)} calls; {dict(c)}")
    print(f"  variant {sum(ms):.0f} ms total (mean {statistics.mean(ms):.2f}, max {max(ms):.1f}); "
          f"A {sum(tms):.0f} ms (mean {statistics.mean(tms):.2f}); select_ip_ic {sum(ipms):.0f} ms over "
          f"{len(ipms)} calls (mean {statistics.mean(ipms) if ipms else 0:.2f})")
    no_sel = sum(1 for r in rs if "select" not in r)
    print(f"  calls that never reached select_ip_ic (cts<=bos, anchor fail, exception in the try): {no_sel}")

print("\nPR12 — as-of resolver picks vs the final drawn pick (same tf / frame start / sid / cycle):")
dk = collections.defaultdict(list)
for d in drawn:
    dk[(d["tf"], d["frame_first_t"], d["sid"], d["cycle"])].append(d)
grp = collections.defaultdict(list)
for r in res:
    grp[(r["caller"], r["tf"], r["frame_first_t"], r["sid"], r["cycle"])].append(r)
tot = diff = nodrawn = 0
for k, rs in sorted(grp.items(), key=lambda kv: str(kv[0])):
    caller, tf, f0, sid, cyc = k
    ds = dk.get((tf, f0, sid, cyc), [])
    final = [(d["bos_t"], d["cts_t"], d["pick_t"], d["rank"]) for d in ds]
    picks = [(r.get("select", {}).get("bos_t"), r.get("select", {}).get("cts_t"), r.get("select", {}).get("pick_t"),
              r.get("select", {}).get("rank")) for r in rs]
    final_pick_ts = {f[2] for f in final}
    n_diff = sum(1 for p in picks if p[2] not in final_pick_ts) if final else 0
    tot += len(picks)
    diff += n_diff
    nodrawn += 0 if final else len(picks)
    if ROWS or (final and n_diff):
        print(f"  {caller} {tf} {f0} sid {sid} cyc {cyc}: {len(picks)} as-of calls, {n_diff} differ from final; "
              f"final(s) {final}")
        if ROWS:
            for p, r in zip(picks, rs):
                print(f"      as-of bos {p[0]} cts {p[1]} -> {p[2]} {p[3]}  (A inners {r['today']} / IP {r['inners']})")
print(f"  total as-of calls {tot}: {diff} differ from a final drawn pick on the same frame; {nodrawn} without a "
      f"drawn record on that frame (probe Phase 2 / a geometry frame != the projection frame)")
