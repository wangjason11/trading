"""PLAN_J §3: after each J2 void, when is cycle T established? Reads `j2c_examples.txt` (j2c_check.py output, cwd).
  PYTHONPATH=TREE python j2_after.py OUT.json   (run for the base, J2 and J2b trees; compare the three JSONs)
"""
import contextlib, importlib.util, io, json, os, sys
import engine_v2
import engine_v2.structure.event_fields as ef
tree = os.path.dirname(os.path.dirname(engine_v2.__file__))
spec = importlib.util.spec_from_file_location("rts", os.path.join(tree, "engine_v2/plans/plan_e_inputs/review_scripts/random_tail_search.py"))
rts = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(rts)
targets = {}
for l in open("j2c_examples.txt"):
    p = l.split()
    targets.setdefault(int(p[1]), {})[int(p[2])] = (int(p[8]), p[-3] == "True")   # apply, beyond
out = {}
for seed, ts in sorted(targets.items()):
    want = set(ts)
    for t, sd, tf, b, rows, nw in rts.wave_trials(seed):
        if t > max(want):
            break
        if t not in want:
            continue
        df = rts._prepare_df(rows)
        ms = rts._make_market_structure(df, cycle0_proximity_blocked=True, cross_fib_limit=rts._NO_P, struct_direction=sd, start_idx=0, structure_id=0, timeframe=tf, pip_size=rts._pip_size_from_pair(df))
        with contextlib.redirect_stdout(io.StringIO()):
            rts._ms_run(ms)
        est = [(int(e.meta["cycle_id"]), int(e.idx), ef.cts_anchor_idx(e)) for e in ms.events if e.type == "CTS_ESTABLISHED"]
        out[f"{seed}/{t}"] = {"apply": ts[t][0], "beyond": ts[t][1], "est": est, "n": len(rows)}
json.dump(out, open(sys.argv[1], "w"))
