"""PLAN_M — the live-`P` census REPORT (PLAN_M §6): reads the `[census]` lines of each window's run.log
(`census_patch.py` + `census_run.py`) and replays the DECIDED rule offline on every COUNTER sub:

- `P` = its first record's parent cycle's BOS threshold — `BOS_CONFIRMED(S, C)` moved by each
  `BOS_THRESHOLD_UPDATED(S, C)`, each known at its LOH M15 moment; every read before the sub's start uses `P` as of
  the start (CL-1: the start folds every move with LOH ≤ start);
- THE LIMIT OBJECT (the `[sweep]` log line): the moves with LOH > start, up to the parent cycle's end; THE EXPORT
  (§7.2, as of the structure's end): `P` at the sub's end (an open sub: the last move) and the moves ≤ its end;
- a cycle T >= 1 established at moment e (on or before the sub's end) is cross-ALLOWED ⟺ no cycle k < T has a
  confirmed CTS price strictly beyond `P` (`beyond`: sub sd +1 → p > P, −1 → p < P) — read with `P(max(e, start))`
  (the live rule, fixed at e), `P(start)` (no live reading) and `P` at the sub's end (what the exported `M` reflects).
  The offline model holds each sub's geometry FIXED (MS's sd-proximity reads `P` too — a cascade is possible on the
  real code) and ignores the IC watch (CL-7).

Prints per window: the H1 facts, every counter sub (start `P`, the object's moves, the export's), the cycles whose
answer the live reading changed (vs the start `P`) or the fix changed (vs `P` at the end), the moves inside a cycle's
PRE-ESTABLISHED window, and the PINNED TABLE of §9 (the expected `[sweep]` / export values on the landed code).

    python census_report.py RUN_LOG [RUN_LOG ...]
"""
import json
import sys


def beyond(p, ref, sd):
    return p > ref if sd == 1 else p < ref


def p_at(sub, t):
    """The live `P` as of M15 moment t (None = the last move up to the parent cycle's end)."""
    p = sub["bos"][0][1]
    for _h1, m15, price in sub["moves"]:
        if t is None or m15 <= t:
            p = price
    return p


def allowed(sub, T, P):
    sd = sub["dir"]
    for k in range(T):
        c = sub["cycles"].get(str(k), {})
        if "conf_price" in c and beyond(c["conf_price"], P, sd):
            return False
    return True


tot = {"windows": 0, "subs": 0, "counter": 0, "counter_moved_in_life": 0, "cycles_in_life": 0,
       "cycles_before_start": 0, "live_vs_start": 0, "live_vs_end": 0, "moves_in_pre_est": 0,
       "bos_checks_failed": 0}
pinned = []
for path in sys.argv[1:]:
    h1, subs, tail = None, [], []
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("[census] h1 "):
            h1 = json.loads(line[len("[census] h1 "):])
        elif line.startswith("[census] sub "):
            subs.append(json.loads(line[len("[census] sub "):]))
        elif line.startswith("[census_run]") or line.startswith("exit ") or "Error" in line:
            tail.append(line.rstrip())
    print(f"=== {path}")
    for t in tail[-4:]:
        print("   ", t)
    if h1 is None:
        print("    NO census lines (crashed before the sweep's end?)")
        continue
    tot["windows"] += 1
    window = tail[0].split("window ")[1].split(":")[0] if tail and "window " in tail[0] else path
    print(f"    H1: {h1['cycles']} cycles, {h1['n_bos']} BOS_CONFIRMED (cycles with != 1: {h1['bos_not_one']}), "
          f"{h1['n_thr']} BOS_THRESHOLD_UPDATED on {len(h1['thr_cycles'])} cycles {h1['thr_cycles']}; "
          f"not strictly further: {h1['thr_not_further']}; at/after the cycle's end: {h1['thr_at_or_after_cycle_end']}")
    tot["subs"] += len(subs)
    for s in subs:
        if not s["counter"]:
            continue
        tot["counter"] += 1
        # §7.3's asserts: one BOS_CONFIRMED, the first step, known by the start.
        ok = len(s["bos"]) == 1 and all(m[0] > s["bos"][0][0] or m[0] == s["bos"][0][0] for m in s["moves"]) \
            and s["bos_m15"][0] <= s["start"]
        tot["bos_checks_failed"] += not ok
        hi = [x for x in (s["end"], s["parent_end_m15"]) if x is not None]
        hi = min(hi) if hi else None
        p0 = p_at(s, s["start"])
        obj_moves = [mv for mv in s["moves"] if mv[1] > s["start"]]
        obj_final = obj_moves[-1][2] if obj_moves else p0
        exp_moves = [mv for mv in obj_moves if hi is None or mv[1] <= hi]
        p_end = exp_moves[-1][2] if exp_moves else p0
        print(f"    sub {s['sub_id']} {'buy' if s['dir'] == 1 else 'sell'} parent {s['parent']} ({s['first_type']}) "
              f"life {s['start']}..{s['end']} ({s['end_reason']}) parent end {s['parent_end_m15']}: P today "
              f"{s['P_now']} {s['P_src']}; BOS {s['bos'][0][1]} -> start {p0}; object moves {len(obj_moves)} -> "
              f"{obj_final}; export (as of the end) moves {len(exp_moves)} -> {p_end}"
              + ("" if ok else "  ** BOS CHECK FAILED **"))
        pinned.append((window, s["sub_id"], p0, len(obj_moves), obj_final, len(exp_moves), p_end))
        if exp_moves:
            tot["counter_moved_in_life"] += 1
            print(f"        moves in life (h1, m15, price): {exp_moves}")
        for T in sorted(int(c) for c in s["cycles"]):
            c = s["cycles"][str(T)]
            if T == 0 or "est" not in c or (s["end"] is not None and c["est"] > s["end"]):
                continue
            if c["est"] <= s["start"]:
                tot["cycles_before_start"] += 1
            else:
                tot["cycles_in_life"] += 1
            a_live = allowed(s, T, p_at(s, max(c["est"], s["start"])))
            a_start, a_end = allowed(s, T, p0), allowed(s, T, p_end)
            prev_conf = s["cycles"].get(str(T - 1), {}).get("conf_at")
            pre = [mv for mv in exp_moves if prev_conf is not None and prev_conf < mv[1] < c["est"]]
            tot["moves_in_pre_est"] += len(pre)
            if a_live != a_start or a_live != a_end or pre:
                tot["live_vs_start"] += a_live != a_start
                tot["live_vs_end"] += a_live != a_end
                print(f"        cycle {T} est {c['est']}: allowed live={a_live} start-P={a_start} end-P={a_end}; "
                      f"moves in its pre-established window {pre}")
print("=== TOTAL", tot)
print("=== PINNED (window, sub_id, P at start, object moves, object final P, export moves, export P)")
for row in pinned:
    print("   ", row)
