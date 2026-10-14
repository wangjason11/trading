"""SHADOW measure of the user's Step 3 (the general ranking when no institutional pattern exists) on every
select_ip_ic call of a full pipeline run — the drawn POI records AND MS's in-flight reads. Read-only: the wrapper
returns the ORIGINAL selection (no behaviour change), exports nothing.

SUPERSEDED since PLAN_P (2026-10-13): its rules include the immediate-unfilled-imbalance rule (IC condition 3
since PLAN_P). Kept as PLAN_L's record.

Run from the repo root: `PYTHONPATH=. python engine_v2/plans/plan_h_inputs/ic_general_ranking_shadow.py > log`.
M15 indices are SUB-LOCAL (the sub's slice frame; `frame0` = its first candle): entity index = local + the slice
offset (2026-10-09 on S_TBU's code: sub 0 +404, sub 1 +1747, sub 2 +2315, sub 6 +3710). Measured: PLAN_H §8.9
"The IC identification revamp".

Step 3 as written (plan_h_inputs/ic_identification_revamp_spec.md): over the IC candidates (every IC condition):
  1. exclude pinbars; 2. the immediate unfilled imbalance (a filter); 3. maru > normal; 4. the most recent IC in
  play before the earliest unfilled sd imbalance after the oldest in play. The ranking rules are their OWN rule set,
  never Step 2's tiebreak rules (PLAN_H §6 ICR-S); this prototype only reads the same per-candle facts.
"""
import inspect
import sys

import engine_v2.zones.poi_zones as pz
from engine_v2.patterns.imbalance import iter_unfilled_imbalances

_orig = pz.select_ip_ic
seen = {}


def _caller():
    for fr in inspect.stack()[2:8]:
        if fr.function in ("select_ip_records", "poi_inner_for_fib"):
            return fr.function
    return "?"


def general_ranking(df, fib_state, config, sel):
    sd = int(fib_state.struct_direction)
    cts = int(fib_state.cts_idx)
    ic = [c for c in sel.candidates if pz.is_ic_candidate(c.touches_zone, c.ic_ref_reason)]
    if not ic:
        return None, "no_ic_candidate", [], None
    ctype = {c.ip.ic_idx: str(df.loc[c.ip.ic_idx, "candle_type"]) for c in ic}
    imm = {c.ip.ic_idx: c.immediate_imbalance_first_c2_idx for c in ic}
    steps = []
    in_play = sorted(i for i in ctype if ctype[i] != "pinbar")
    steps.append(("exclude_pinbar", list(in_play)))
    if not in_play:
        return None, "all_pinbar", steps, None
    if len(in_play) == 1:
        return in_play[0], "exclude_pinbar", steps, None
    passing = [i for i in in_play if imm[i] is not None]
    steps.append(("immediate", list(passing)))
    if len(passing) == 1:
        return passing[0], "immediate_unfilled_imbalance", steps, imm[passing[0]]
    in_play = passing or in_play
    best = min(("maru", "normal").index(ctype[i]) for i in in_play)
    by_type = [i for i in in_play if ("maru", "normal").index(ctype[i]) == best]
    steps.append(("type", list(by_type)))
    if len(by_type) == 1:
        return by_type[0], "ic_candle_type", steps, None
    oldest = min(by_type)
    ref = min(iter_unfilled_imbalances(df, oldest + 1, cts, cts, config.fill_threshold, direction=sd,
                                       evaluated_at=None), key=lambda inst: inst.start_idx)
    pick = max(i for i in by_type if i < ref.start_idx)
    return pick, "most_recent_before_imbalance", steps, int(ref.start_idx)


def _wrapped(df, fib_state, config, **kw):
    sel = _orig(df, fib_state, config, **kw)
    if sel.pick is not None or fib_state is None or fib_state.fib is None:
        return sel
    if not sel.candidates and not (fib_state.active or fib_state.locked):
        return sel
    who = _caller()
    t0 = df.index[0]
    key = (df.attrs.get("timeframe"), str(df.loc[t0, "time"]) if "time" in df.columns else t0,
           fib_state.structure_id, getattr(fib_state, "cycle_id", None), int(fib_state.bos_idx),
           int(fib_state.cts_idx), who)
    if key in seen:
        return sel
    pick, why, steps, c2 = general_ranking(df, fib_state, config, sel)
    seen[key] = (pick, why)
    ctypes = {c.ip.ic_idx: str(df.loc[c.ip.ic_idx, "candle_type"]) for c in sel.candidates
              if pz.is_ic_candidate(c.touches_zone, c.ic_ref_reason)}
    print(f"\n== {key[0]} frame0={key[1]} sid={key[2]} cyc={key[3]} bos={key[4]} cts={key[5]}"
          f" sd={int(fib_state.struct_direction)} via={who}  STEP3 pick={pick} by={why} ref_c2={c2}")
    print(f"   IC candidates: {sorted(ctypes.items())}")
    for name, ids in steps:
        print(f"   after {name}: {ids}")
    if pick is not None:
        print(f"   pick h={float(df.loc[pick, 'h']):.5f} l={float(df.loc[pick, 'l']):.5f}")
    sys.stdout.flush()
    return sel


pz.select_ip_ic = _wrapped

if __name__ == "__main__":
    import engine_v2.run_replay as rr
    _orig_run = rr.run_pipeline

    class _Stop(Exception):
        pass

    def _run_and_stop(*a, **k):
        _orig_run(*a, **k)
        raise _Stop()

    rr.run_pipeline = _run_and_stop
    rr.export_csv = lambda *a, **k: None
    try:
        rr.main()
    except _Stop:
        print("\n[shadow] pipeline done; no-IP selections seen:", len(seen))
        for k, v in seen.items():
            print("  ", k, "->", v)
