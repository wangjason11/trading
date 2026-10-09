"""PLAN_J §3: every J2 void (the trials whose `sig` differs between the base and the J2 corpus) — does its APPLY candle close
beyond `cts_threshold`? (the J2-Q1 (c) test).  cd WORKDIR && PYTHONPATH=J2_TREE python j2c_check.py BASE_SEARCH_DIR J2_SEARCH_DIR
Prints one EX line per void (seed, trial, tf, sd, anchor, apply, pattern, threshold, apply close, beyond).
"""
import contextlib, glob, importlib.util, io, json, os, sys
import engine_v2
import engine_v2.structure.market_structure as msm
tree = os.path.dirname(os.path.dirname(engine_v2.__file__))
spec = importlib.util.spec_from_file_location("rts", os.path.join(tree, "engine_v2/plans/plan_e_inputs/review_scripts/random_tail_search.py"))
rts = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(rts)
B, V = sys.argv[1], sys.argv[2]
diff = []
for f in glob.glob(os.path.join(V, "W*_*.jsonl")):
    name = os.path.basename(f)
    base = {json.loads(l)["t"]: json.loads(l) for l in open(os.path.join(B, name))}
    for l in open(f):
        r = json.loads(l)
        if base[r["t"]].get("sig") != r.get("sig"):
            diff.append((int(name.split("_")[1][:-6]), r["t"]))
from collections import Counter
res = Counter()
gens = {}
for seed, t in sorted(diff):
    for tt, sd, tf, b, rows, nw in rts.wave_trials(seed):
        if tt == t:
            break
    msm.PJ_LOG.clear()
    df = rts._prepare_df(rows)
    ms = rts._make_market_structure(df, cross_fib_limit=rts._NO_P, struct_direction=sd, start_idx=0, structure_id=0, timeframe=tf, pip_size=rts._pip_size_from_pair(df))
    with contextlib.redirect_stdout(io.StringIO()):
        rts._ms_run(ms)
    for L in msm.PJ_LOG:
        if L["kind"] == "J2" and L["void"]:
            c = rows[L["apply"]]["c"]; th = L["cts_th"]
            beyond = c > th if sd == 1 else c < th
            res["apply close beyond" if beyond else "apply close NOT beyond"] += 1
            print("EX", seed, t, tf, sd, "anchor", L["anchor"], "apply", L["apply"], "pat", L["pat"], "th", th, "apply close", c, "beyond", beyond, "next_cycle", L["cycle_next"])
print(len(diff), "differing trials;", dict(res))
