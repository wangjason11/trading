"""`hover_sim.py` + the A2 candidate (a candle shows its structure's state after the LAST-replayed event stamped at
or before it) — the PLAN_N hover follow-up's third option (2026-10-12; not chosen). From the repo root:

    PYTHONPATH=. python engine_v2/plans/plan_n_inputs/hover_sim_a2.py SAVE_DIR
"""
import runpy, sys
g = runpy.run_path(sys.argv[0].replace("hover_sim_a2.py", "hover_sim.py"), run_name="sim")
order, idx_set, _apply, points, report = g["order"], g["idx_set"], g["_apply"], g["points"], g["report"]

def a2():
    open_ranges, snaps, w = {}, {}, 0
    for ev in order:
        sid, sd, idx = int(ev.meta.get("structure_id", 0)), int(ev.meta.get("struct_direction", 1)), int(ev.idx)
        _apply(open_ranges, ev, sid, sd, idx)
        r = open_ranges.get(sid)
        w += 1
        snaps.setdefault(sid, []).append((idx, w, dict(r) if r is not None else None))
    rbc = {}
    for sid, sn in snaps.items():
        for c in sorted(idx_set):
            cand = [s for s in sn if s[0] <= c]
            if cand:
                st = max(cand, key=lambda s: s[1])[2]
                if st is not None:
                    rbc.setdefault(c, {})[sid] = st
    return rbc

report("A2 (per sid, the last-replayed event stamped at or before the candle)", points(a2()))
