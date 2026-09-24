# RUNS AGAINST 754a642 (pre-Plan-F) ONLY: it calls has_unfilled_imbalance without evaluated_at,
# which Plan F made a required keyword. Kept as the measurement record of PLAN_F_inputs.md §2.
"""Cascade measurement: APPLY the imbalance c3-knowability rule per scope, run the replay.

Scopes (env C3_SCOPES, comma list):
  sweep  - POI activation sweep enters an instance at start_idx+1 (R1) instead of start_idx
  ms     - MarketStructure in-flight reads (cycle-0 snapshot market_structure.py:2050, and the
           in-flight POI-inner resolver compute_poi_inners_for_cycle -> find_ic_candidates /
           select_fib_anchor_for_cycle): knowable_at = the MS evaluation moment
           (pattern path: _apply_pattern_at_apply_idx.apply_idx; raw path: _maybe_update_cts_pre_confirm.i)
  fib    - downstream FibTracker / cross_cycle_fib reads: knowable_at = the driving event's moment
           (CTS_ESTABLISHED -> meta confirmed_at; any other event -> ev.idx)
  poiid  - derive_poi_zones -> find_ic_candidates (retrospective IC identification):
           knowable_at = check_to_idx (the final fib's cts_idx)
R1-exact filter: an instance is visible at K iff start_idx < K; its overlap test uses
[start_idx, min(end_idx, K-1)] (the knowable prefix). Fill check unchanged (is_filled(check_to_idx)).
Logs every call where the moment could not be found, and every call the rule changed.
"""
import collections
import json
import os
import runpy
import sys
import textwrap
import inspect
import atexit

REPO = os.getcwd()
sys.path.insert(0, REPO)
SCOPES = set(s for s in os.environ.get("C3_SCOPES", "").split(",") if s)
OUT = os.environ.get("C3_OUT", "c3_changes.jsonl")

import engine_v2.patterns.imbalance as imb_mod
import engine_v2.structure.market_structure as ms_mod
import engine_v2.zones.fib_tracker as fib_mod
import engine_v2.zones.poi_zones as poi_mod
import engine_v2.zones.cross_cycle_fib as ccf_mod

_orig_has = imb_mod.has_unfilled_imbalance
_orig_sweep = poi_mod._compute_poi_activation_history
stats = collections.Counter()
_fh = open(OUT, "w", encoding="utf-8")


def _frames(skip, depth=40):
    f = sys._getframe(skip)
    out = []
    while f is not None and len(out) < depth:
        out.append(f)
        f = f.f_back
    return out


def _scope_and_moment():
    """Return (scope, knowable_at or None, how)."""
    frames = _frames(3)
    names = [fr.f_code.co_name for fr in frames]
    in_ms = any(n in ("_refresh_poi_inners_for_cycle", "_update_cycle0_data") for n in names)
    if in_ms:
        for fr in frames:
            n = fr.f_code.co_name
            if n == "_apply_pattern_at_apply_idx":
                return "ms", int(fr.f_locals["apply_idx"]), "apply_idx"
            if n == "_maybe_update_cts_pre_confirm":
                return "ms", int(fr.f_locals["i"]), "raw_i"
        return "ms", None, "ms-no-moment"
    if "derive_poi_zones" in names:
        return "poiid", None, "check_to"
    # downstream fib / cross-cycle: nearest StructureEvent in the stack
    for fr in frames:
        for k, v in list(fr.f_locals.items()):
            if type(v).__name__ == "StructureEvent":
                if v.type == "CTS_ESTABLISHED":
                    return "fib", int(v.meta["confirmed_at"]), "EST.confirmed_at"
                return "fib", int(v.idx), f"{v.type}.idx"
    return "other", None, "no-event"


def _has_r1(df, start_idx, end_idx, check_to_idx, fill_threshold, direction, K):
    for inst in df.attrs.get("imbalances", []):
        if direction is not None and inst.direction != direction:
            continue
        if inst.start_idx >= K:
            continue  # first c3 not closed at K: not knowable
        vis_end = min(inst.end_idx, K - 1)
        if not (inst.start_idx <= end_idx and vis_end >= start_idx):
            continue
        if not inst.is_filled(df, check_to_idx, fill_threshold):
            return True
    return False


def c3_has(df, start_idx, end_idx, check_to_idx, fill_threshold=0.70, *, direction=None):
    orig = _orig_has(df, start_idx, end_idx, check_to_idx, fill_threshold, direction=direction)
    scope, K, how = _scope_and_moment()
    stats[f"calls {scope} via {how}"] += 1
    if scope not in SCOPES:
        return orig
    if K is None:
        K = int(check_to_idx)
    if K != int(check_to_idx):
        stats[f"K!=check_to {scope} via {how} (K-check={K - int(check_to_idx)})"] += 1
    new = _has_r1(df, start_idx, end_idx, check_to_idx, fill_threshold, direction, K)
    if new != orig:
        stats[f"CHANGED {scope} via {how}"] += 1
        site = None
        for fr in _frames(2, 3):
            fn = fr.f_code.co_filename.replace("\\", "/")
            if "/engine_v2/" in fn:
                site = f"{fn.split('/engine_v2/')[1]}:{fr.f_lineno}"
                break
        _fh.write(json.dumps({"scope": scope, "how": how, "site": site, "K": K,
                              "check_to": int(check_to_idx), "window": [int(start_idx), int(end_idx)],
                              "orig": orig, "new": new, "n": len(df)}) + "\n")
    return new


def _make_sweep_r1():
    src = textwrap.dedent(inspect.getsource(_orig_sweep))
    needle = "enter_idx = max(inst.start_idx, first_active)"
    assert src.count(needle) == 1
    src = src.replace(needle, "enter_idx = max(inst.start_idx + 1, first_active)")
    ns = dict(vars(poi_mod))
    exec(compile(src, "<sweep-r1>", "exec"), ns)
    return ns["_compute_poi_activation_history"]


if "sweep" in SCOPES:
    poi_mod._compute_poi_activation_history = _make_sweep_r1()

for mod in (imb_mod, ms_mod, fib_mod, poi_mod, ccf_mod):
    if hasattr(mod, "has_unfilled_imbalance"):
        mod.has_unfilled_imbalance = c3_has


@atexit.register
def _summary():
    _fh.close()
    print(f"\n=== C3 VARIANT SUMMARY scopes={sorted(SCOPES)} ===")
    for k in sorted(stats):
        print(f"{stats[k]:7d}  {k}")


sys.argv = ["engine_v2.run_replay"]
runpy.run_module("engine_v2.run_replay", run_name="__main__", alter_sys=True)
