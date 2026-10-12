"""PLAN_M landing 2 — the census REPORT on the LANDED code (§8 / §9): the engine's own values against the pinned table.

    python census_l2_report.py PINNED_TXT "PRE_LOG L2_LOG" ["PRE_LOG L2_LOG" ...]

PINNED_TXT = `census_pre_build.txt` (its `=== PINNED` rows, `census_report.py`'s offline model). Each pair = one
window's pre-build run.log (a `git archive` of 27ab6da, `census_patch.py`) and its L2 run.log (the landed code +
`census_patch.py`, `census_run.py TREE --export`). Per window:

- every counter sub's row FROM THE ENGINE — `P` at start / the object's moves → its last `P` (the `[census] sub`
  line: the limit object) / the export's moves → its `P` (the `[census] export` lines: the mirrored fib records,
  as of the sub's end) — vs the pinned row: MATCH / DIFF;
- the R6-2 tripwire line and every `WARNING [cross_fib_limit]`;
- CL-7's other side (§3 "readings"): a counter sub's exported IC break (`cross_limit_anchor`) confirmed BEFORE the
  move that set its exported `P` — counted and listed;
- 2025-02-01's sub 6 cycle 4: its fib records' `via` (pinned: not `cross_fib_limit`);
- the knock-on (§1 item 1 — listed, no prediction): every sub whose `[census] sub` structure facts (its cycles'
  establishment moments / confirmed CTS prices, life) or non-counter `P` differ from the pre-build run.
"""
import ast
import json
import sys


def read(path):
    h1, subs, exports, trip, warns, window = None, {}, [], None, [], None
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("[census] h1 "):
            h1 = json.loads(line[len("[census] h1 "):])
        elif line.startswith("[census] sub "):
            s = json.loads(line[len("[census] sub "):])
            subs[s["sub_id"]] = s
        elif line.startswith("[census] export "):
            exports.append(json.loads(line[len("[census] export "):]))
        elif line.startswith("[cross_fib_limit] tripwire:"):
            trip = line.strip()
        elif line.startswith("WARNING [cross_fib_limit]"):
            warns.append(line.strip())
        elif line.startswith("[census_run] window "):
            window = line.split("window ")[1].split(":")[0]
    return dict(h1=h1, subs=subs, exports=exports, trip=trip, warns=warns, window=window)


pinned = {}
in_pin = False
for line in open(sys.argv[1], encoding="utf-8"):
    if line.startswith("=== PINNED"):
        in_pin = True
        continue
    if in_pin and line.strip().startswith("("):
        row = ast.literal_eval(line.strip())
        pinned[(row[0], row[1])] = row

tot = dict(windows=0, counter=0, match=0, diff=0, no_export=0, tripwire_mismatches=0, cl7_before_move=0,
           knock_on_subs=0)
for pair in sys.argv[2:]:
    pre_path, l2_path = pair.split()
    pre, l2 = read(pre_path), read(l2_path)
    tot["windows"] += 1
    w = l2["window"]
    print(f"=== window {w}  ({l2_path})")
    print(f"    {l2['trip']}")
    if l2["trip"] and not l2["trip"].endswith(" 0 mismatches"):
        tot["tripwire_mismatches"] += int(l2["trip"].split(", ")[1].split()[0])
    for wl in l2["warns"]:
        print(f"    {wl}")
    by_sub = {}
    for e in l2["exports"]:
        by_sub.setdefault(e["sub_id"], []).append(e)
    for sid, s in sorted(l2["subs"].items()):
        if not s["counter"]:
            continue
        tot["counter"] += 1
        moves = s.get("P_moves") or []
        p_start = s["P_start"] if moves else s["P_now"]
        recs = by_sub.get(sid, [])
        keys = {(e["P"], e["P_start"], e["moves"]) for e in recs}
        assert len(keys) <= 1, (w, sid, keys)                     # one limit per structure
        if keys:
            (ep, _eps, em), = keys
            exp = (em or 0, ep)
        else:
            exp = (None, None)
            tot["no_export"] += 1
        row = (w, sid, p_start, len(moves), s["P_now"], exp[0], exp[1])
        pin = pinned.get((w, sid))
        # the SOURCE too (landing 2 review B, MINOR-4): the object's and every exported record's
        src_ok = s["P_src"] == "parent_bos_threshold" and all(e["src"] == "parent_bos_threshold" for e in recs)
        ok = (pin is not None and src_ok and tuple(pin[2:5]) == row[2:5]
              and (exp == (None, None) or tuple(pin[5:]) == row[5:]))
        tot["match" if ok else "diff"] += 1
        print(f"    sub {sid} {s['P_src']}: engine {row[2:]}  pinned {None if pin is None else pin[2:]}  "
              f"{'MATCH' if ok else 'DIFF'}" + ("" if recs else "  (no fib record: export unseen)"))
        # CL-7's other side: the exported break of P(end) confirmed before the move that set P(end)
        end = s["end"]
        known = [mv for mv in moves if end is None or mv[0] <= end]
        if known:
            last_at = known[-1][0]
            for e in recs:
                if e["ic_src"] == "cross_limit_anchor" and e["ic_brk"] is not None and int(e["ic_brk"]) < last_at:
                    tot["cl7_before_move"] += 1
                    print(f"        CL-7: cycle {e['cycle']} v{e['version']} break confirmed {e['ic_brk']} < the move "
                          f"to {known[-1][1]} known at {last_at}")
    if w.startswith("2025-01-28"):
        for e in sorted(by_sub.get(6, []), key=lambda e: (e["cycle"], str(e["version"]))):
            if e["cycle"] == 4:
                print(f"    sub 6 cycle 4 record ({e['lens']}, v{e['version']}, cross={e['cross']}): via={e['via']}")
    # the knock-on: structure facts / a non-counter P changed vs the pre-build run
    for sid in sorted(set(pre["subs"]) | set(l2["subs"])):
        a, b = pre["subs"].get(sid), l2["subs"].get(sid)
        if a is None or b is None:
            tot["knock_on_subs"] += 1
            print(f"    KNOCK-ON sub {sid}: present only {'pre-build' if b is None else 'L2'}")
            continue
        facts = ("dir", "parent", "first_type", "start", "end", "end_reason", "cycles", "n_records")
        diff = [k for k in facts if a[k] != b[k]]
        if not b["counter"] and (a["P_now"], a["P_src"]) != (b["P_now"], b["P_src"]):
            diff.append("P")
        if diff:
            tot["knock_on_subs"] += 1
            print(f"    KNOCK-ON sub {sid}: {diff}")
    if pre["h1"] != l2["h1"]:
        print("    KNOCK-ON: the H1 facts differ")
print("=== TOTAL", tot)
