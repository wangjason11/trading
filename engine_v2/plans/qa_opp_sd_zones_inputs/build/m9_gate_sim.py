"""PLAN_R step 3's reference simulator: the user's round-15 GATE (a reference update that would leave the proximity
reference gap <= 40 is skipped; no counters) + the five adopted recommendations, on m4_sim's inputs (H1 frames, the H1
parent cycles, S_Q's sub zones + (1,3230) from shadow_full2). Independent of the engine. Also carries the round-14
source switch ("Contributed + reachable") that m4_sim predates.

The rules simulated (PLAN_R §9 as drafted):
  - the scan starts with each side's reference = its LAST rung (parent BOS / CTS); every other rung is an INWARD
    update from there (rec 4: the parent POI is gated too);
  - per candle, phase A: a side whose reference was PASSED moves to the closest unpassed rung that passes the gate
    (against the other side's reference as it stands); phase B (rec 3: after every outward move): a side not forced
    looks at its unpassed rungs strictly INSIDE its current reference, closest first, and takes the first that passes
    the gate against the other side's (updated) reference (rec 2); none -> the reference stays (the user's "remains
    what it was");
  - a SUPERSEDED reference (unpassed, no longer on its ladder) stays unless the ladder's natural pick (the closest
    unpassed rung) is outward of it (an outward update is never blocked) or a rung inside it passes the gate (rec 1);
    it stays pass-testable;
  - the exception (rec 5): parent BOS-CTS inner gap <= 40 -> the references stay on the parent BOS / CTS, at most one
    sd + one opp_sd trigger;
  - lock / pass / tie order as step 2 (m4_sim); the opp_sd unlock not modelled (logged if an opp_sd lock occurs).
Logged for the open questions: a candle where BOTH sides have inward candidates (phase-B order could matter), a
superseded reference whose natural pick is outward (moved), a forced move whose closest candidate is inward of the
passed reference."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m4_sim as base  # noqa: E402  (loads frames, CY, ZONES, REC)

C, H, L = base.C, base.H, base.L
PIP, PROX, GATE = base.PIP, base.PROX, base.NARROW
rnd = base.rnd

# The subs that only 1045 / 1046 create (step 2) — absent unless the sim fires those triggers.
STEP2_ONLY = {(1, 2): {(-1, 4179): (1045, "opp"), (1, 4183): (1046, "sd")}}
REC = {k: {s: v for s, v in d.items() if s not in STEP2_ONLY.get(k, {})} for k, d in base.REC.items()}
START = {}
for d in base.REC.values():
    for s, v in d.items():
        START[s] = min(START.get(s, v), v)


def run(key, overlay=True, verbose=False):
    cy = base.CY[key]; sd = cy["sd"]; lo, hi = cy["lo"], cy["hi"]
    bos_in = cy["bos"][0]; cts_in = cy["cts"][0]

    def psteps(steps, side):
        k = "bottom" if side == "buy" else "top"
        return [(int(s.get("expanded_idx", s.get("base_idx"))), float(s[k])) for s in steps]
    P = []
    poi_inner = None
    for (top, bot, side) in cy["pois"]:
        inner = top if side == "buy" else bot; outer = bot if side == "buy" else top
        if sd * (inner - bos_in) > 0:
            P.append(dict(label="P.POI", side="sd", inner=inner, steps=[(0, outer)], last=False, par=True, kind="POI"))
            poi_inner = inner
    P.append(dict(label="P.BOS", side="sd", inner=bos_in, steps=psteps(cy["bos"][3], cy["bos"][2]), last=True,
                  par=True, kind="BOS"))
    P.append(dict(label="P.CTS", side="opp", inner=cts_in, steps=psteps(cy["cts"][3], cy["cts"][2]), last=True,
                  par=True, kind="CTS"))
    LAST = {s: [p for p in P if p["side"] == s and p["last"]][0] for s in ("sd", "opp")}
    toward = {"sd": sd, "opp": -sd}           # from a side's zones toward the other side

    def closer(side, a, b):                   # a strictly closer to the other side than b
        return toward[side] * (a - b) > 0

    def gap(sd_inner, opp_inner):
        return rnd(sd * (opp_inner - sd_inner) / PIP)

    def ok(side, r, other):                   # the gate: the gap AFTER the update stays above the threshold
        g = gap(r["inner"], other["inner"]) if side == "sd" else gap(other["inner"], r["inner"])
        return g > GATE

    def beyond_outer(side, close, outer):
        if side == "sd":
            return close < outer if sd == 1 else close > outer
        return close > outer if sd == 1 else close < outer

    def h1_outer(z, h):
        o = z["steps"][0][1]
        for f, v in z["steps"]:
            if f <= h:
                o = v
        return o

    rec = REC.get(key, {}) if overlay else {}

    def sub_side(k):
        return "sd" if k[0] == sd else "opp"

    def qualifies(side, z):
        return side == "opp" or (poi_inner is not None and sd * (z["inner"] - poi_inner) < 0)

    def usable(k, z, h):
        return z["moment"] <= 4 * h - 1 and rec[k] <= 4 * h - 1

    def zid(k, z):
        return f"{k}c{z['cyc']}.{z['kind']}"

    def contributed(side, k, h):              # the strict cycle (most recent usable BOS) and its rung-able zones
        uz = [z for z in base.ZONES[k] if usable(k, z, h)]
        bc = [z["cyc"] for z in uz if z["kind"] == "BOS"]
        if not bc:
            return []
        mc = max(bc)
        return [z for z in sorted(uz, key=lambda z: z["kind"] != "BOS") if z["cyc"] == mc and qualifies(side, z)]

    def reachable(side, z):
        return toward[side] * (z["inner"] - LAST[side]["inner"]) > 0

    exception = gap(bos_in, cts_in) <= GATE
    passed = set(); lock = {"sd": False, "opp": False}; source = {"sd": None, "opp": None}
    ref = {"sd": LAST["sd"], "opp": LAST["opp"]}
    fired = {"sd": 0, "opp": 0}
    expected = "sd"; out = []; log = []; flags = []
    log.append(f"  scan [{lo},{hi}] sd={sd} BOS-CTS inner gap {gap(bos_in, cts_in)}"
               f"{' EXCEPTION' if exception else ''}")
    for h in range(lo, hi + 1):
        rungs = {}
        for side in ("sd", "opp"):
            cands = [k for k in rec if sub_side(k) == side and rec[k] <= 4 * h - 1]
            cur = source[side]
            newer = [k for k in cands if cur is None or (START[k], k) > (START[cur], cur)]
            elig = [k for k in newer if any(reachable(side, z) for z in contributed(side, k, h))]
            if elig:
                source[side] = max(elig, key=lambda k: (START[k], k))
                if source[side] != cur:
                    log.append(f"  {h}: {side} source -> {source[side]} (was {cur})")
            zs = []
            if source[side] is not None:
                k = source[side]
                for z in contributed(side, k, h):
                    zs.append(dict(label=zid(k, z), side=side, inner=z["inner"], z=z, last=False, par=False,
                                   kind=z["kind"]))
            rungs[side] = [p for p in P if p["side"] == side] + zs
        prev = dict(ref)

        def order(side):
            avail = [r for r in rungs[side] if r["label"] not in passed]
            avail.sort(key=lambda r: (0 if r["par"] else 1, 0 if r["kind"] in ("BOS", "CTS") else 1))
            res = []                               # closest first; ties keep the tie order (stable)
            for r in avail:
                i = 0
                while i < len(res) and not closer(side, r["inner"], res[i]["inner"]):
                    i += 1
                res.insert(i, r)
            return res

        if exception:
            new = {"sd": LAST["sd"], "opp": LAST["opp"]}
        else:
            new = dict(ref)
            forced = [s for s in ("sd", "opp") if ref[s]["label"] in passed]
            # phase A: forced (passed) sides
            for side in forced:
                other = new["opp" if side == "sd" else "sd"]
                cand = order(side)
                if cand and closer(side, cand[0]["inner"], ref[side]["inner"]):
                    flags.append(f"{key} {h}: forced {side} move, closest candidate {cand[0]['label']} is INSIDE the "
                                 f"passed {ref[side]['label']}")
                pick = next((r for r in cand if ok(side, r, other)), None)
                assert pick is not None, f"{key} {h}: forced {side} move with no candidate passing the gate"
                if pick["last"]:
                    lock[side] = True
                new[side] = pick
            # phase B: the other sides, inward candidates against the updated other side
            inward_sides = []
            for side in ("sd", "opp"):
                if side in forced or lock[side]:
                    continue
                cand = order(side)
                in_ladder = any(r["label"] == ref[side]["label"] for r in rungs[side])
                if not in_ladder and cand and not closer(side, cand[0]["inner"], ref[side]["inner"]):
                    flags.append(f"{key} {h}: {side} reference {ref[side]['label']} superseded; natural pick "
                                 f"{cand[0]['label']} is OUTWARD -> moved")
                    new[side] = cand[0]
                    continue
                inside = [r for r in cand if closer(side, r["inner"], ref[side]["inner"])]
                if inside:
                    inward_sides.append((side, inside))
            if len(inward_sides) == 2:
                flags.append(f"{key} {h}: BOTH sides have inward candidates: "
                             + "; ".join(f"{s} {[r['label'] for r in c]}" for s, c in inward_sides))
            for side, inside in inward_sides:
                other = new["opp" if side == "sd" else "sd"]
                pick = next((r for r in inside if ok(side, r, other)), None)
                skipped = [r["label"] for r in inside[: inside.index(pick) if pick else len(inside)]]
                if skipped:
                    log.append(f"  {h}: {side} GATE skips {skipped} (vs {other['label']} {other['inner']})")
                if pick is not None:
                    new[side] = pick
        ref = new
        g = gap(ref["sd"]["inner"], ref["opp"]["inner"])
        if any(prev[s]["label"] != ref[s]["label"] for s in ("sd", "opp")) or h == lo:
            log.append(f"  {h}: sd={ref['sd']['label']}({ref['sd']['inner']}) opp={ref['opp']['label']}"
                       f"({ref['opp']['inner']}) gap={g} lock={lock}")
        z = ref[expected]; side = expected
        if side == "sd":
            hit = (L[h] <= z["inner"] + PROX * PIP) if sd == 1 else (H[h] >= z["inner"] - PROX * PIP)
        else:
            hit = (H[h] >= z["inner"] - PROX * PIP) if sd == 1 else (L[h] <= z["inner"] + PROX * PIP)
        if hit:
            if exception and fired[side] >= 1:
                log.append(f"  {h}: {side} BLOCKED (the exception) on {z['label']}")
            else:
                fired[side] += 1
                out.append((h, side, z["label"], z["inner"]))
                log.append(f"  {h}: TRIGGER {side} {z['label']} {z['inner']} (l={L[h]} h={H[h]} c={C[h]}) gap={g}")
                expected = "opp" if expected == "sd" else "sd"
                if key in STEP2_ONLY and (h, side) in STEP2_ONLY[key].values():
                    flags.append(f"{key} {h}: the sim FIRED a step-2-only trigger ({side}) — its sub is missing here")
        # pass test on close(h): every unpassed non-last rung on the ladder + a kept superseded reference
        for side in ("sd", "opp"):
            test = [r for r in rungs[side] if not r["last"] and r["label"] not in passed]
            if not ref[side]["last"] and ref[side]["label"] not in passed and \
                    all(r["label"] != ref[side]["label"] for r in test):
                test.append(ref[side])
            for r in test:
                o = h1_outer(r, h) if r["par"] else base.outer_at(r["z"], 4 * h + 3)
                if beyond_outer(side, C[h], o):
                    passed.add(r["label"])
                    log.append(f"  {h}: close {C[h]} beyond {r['label']} outer {o} -> passed from {h + 1}")
                    if side == "opp" and r is ref[side]:
                        flags.append(f"{key} {h}: the opp_sd REFERENCE passed (a lock may follow; unlock not modelled)")
    if verbose:
        print("\n".join(log))
    return out, flags


if __name__ == "__main__":
    allflags = []
    print("=== parent only (the batch: collapsed cycles + no overlay) ===")
    for k in sorted(base.CY):
        o, f = run(k, overlay=False, verbose=True)
        print(k, o); allflags += f
    print("=== overlay ===")
    for k in ((0, 1), (1, 2)):
        o, f = run(k, overlay=True, verbose=True)
        print(k, o); allflags += f
    print("=== flags ===")
    print("\n".join(allflags) if allflags else "(none)")
