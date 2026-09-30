# RUNS AGAINST 754a642 (pre-Plan-F) ONLY: it calls has_unfilled_imbalance without evaluated_at,
# which Plan F made a required keyword. Kept as the measurement record of PLAN_F_inputs.md §2.
"""Shadow measurement: imbalance c3-knowability, every call site, one replay.

Wraps has_unfilled_imbalance (in every importing module) and the POI sweep
_compute_poi_activation_history. Each wrapper RETURNS THE ORIGINAL RESULT
(the replay's outputs are byte-identical to the current code) and logs every
call where excluding not-yet-knowable instances would change the answer:
  R1: exclude inst with inst.start_idx + 1 > check_to_idx (first c3 not closed)
  R2: exclude inst with inst.end_idx   + 1 > check_to_idx (last c3 not closed)
POI sweep: enter at start_idx+1 (R1) / end_idx+1 (R2) instead of start_idx.

Run from the repo root:  python <this file>  > shadow_run.log 2>&1
Output: $SHADOW_OUT (jsonl) + a per-site call/flip count summary at exit.
"""
import atexit
import collections
import inspect
import json
import os
import runpy
import sys
import textwrap

REPO = os.getcwd()
sys.path.insert(0, REPO)
OUT = os.environ.get("SHADOW_OUT", "shadow_flips.jsonl")

import engine_v2.patterns.imbalance as imb_mod
import engine_v2.structure.market_structure as ms_mod
import engine_v2.zones.fib_tracker as fib_mod
import engine_v2.zones.poi_zones as poi_mod
import engine_v2.zones.cross_cycle_fib as ccf_mod

_orig_has = imb_mod.has_unfilled_imbalance
_orig_sweep = poi_mod._compute_poi_activation_history

calls = collections.Counter()
flips = collections.Counter()
_fh = open(OUT, "w", encoding="utf-8")

INTS = ("i", "apply_idx", "current_candle", "t", "cur_idx", "confirm_idx", "idx",
        "cts_idx", "bos_idx", "own_imb_start", "sid", "cycle_id", "target_cycle")


def _entity(df):
    info = {"n": int(len(df))}
    if "time" in df.columns and len(df) > 1:
        t0 = df["time"].iloc[0]
        t1 = df["time"].iloc[1]
        info["t0"] = str(t0)
        try:
            info["tf_min"] = int((t1 - t0).total_seconds() // 60)
        except Exception:
            info["tf_min"] = None
    return info


def _time_at(df, idx):
    try:
        if "time" in df.columns and 0 <= idx < len(df):
            return str(df["time"].iloc[idx])
    except Exception:
        pass
    return None


def _frames(skip, depth):
    f = sys._getframe(skip)
    out = []
    while f is not None and len(out) < depth:
        out.append(f)
        f = f.f_back
    return out


def _context(skip=2, depth=18):
    site = None
    chain = []
    event = None
    ints = {}
    for fr in _frames(skip + 1, depth):
        fn = os.path.relpath(fr.f_code.co_filename, REPO).replace("\\", "/")
        if not fn.startswith("engine_v2"):
            continue
        func = fr.f_code.co_name
        if site is None:
            site = f"{fn}:{fr.f_lineno}"
        chain.append(f"{os.path.basename(fn)}:{func}:{fr.f_lineno}")
        loc = fr.f_locals
        if event is None:
            for k, v in list(loc.items()):
                if type(v).__name__ == "StructureEvent":
                    m = getattr(v, "meta", {}) or {}
                    event = {"var": k, "fn": func, "type": v.type, "idx": int(v.idx),
                             "confirmed_at": m.get("confirmed_at"),
                             "sid": m.get("structure_id"), "cycle": m.get("cycle_id")}
                    break
        for k in INTS:
            if k in loc and f"{func}.{k}" not in ints:
                v = loc[k]
                if isinstance(v, int) and not isinstance(v, bool):
                    ints[f"{func}.{k}"] = int(v)
    return site, chain, event, ints


def _site_only(skip=2):
    for fr in _frames(skip + 1, 6):
        fn = os.path.relpath(fr.f_code.co_filename, REPO).replace("\\", "/")
        if fn.startswith("engine_v2"):
            return f"{fn}:{fr.f_lineno}"
    return "?"


def _eval(df, start_idx, end_idx, check_to_idx, fill_threshold, direction, rule):
    unknowable = []
    result = False
    for inst in df.attrs.get("imbalances", []):
        if direction is not None and inst.direction != direction:
            continue
        if not inst.overlaps(start_idx, end_idx):
            continue
        if rule == "R1" and inst.start_idx + 1 > check_to_idx:
            unknowable.append((inst.start_idx, inst.end_idx, inst.direction))
            continue
        if rule == "R2" and inst.end_idx + 1 > check_to_idx:
            unknowable.append((inst.start_idx, inst.end_idx, inst.direction))
            continue
        if not inst.is_filled(df, check_to_idx, fill_threshold):
            result = True
    return result, unknowable


def shadow_has(df, start_idx, end_idx, check_to_idx, fill_threshold=0.70, *, direction=None):
    orig = _orig_has(df, start_idx, end_idx, check_to_idx, fill_threshold, direction=direction)
    r1, unk1 = _eval(df, start_idx, end_idx, check_to_idx, fill_threshold, direction, "R1")
    r2, unk2 = _eval(df, start_idx, end_idx, check_to_idx, fill_threshold, direction, "R2")
    site = _site_only()
    calls[site] += 1
    if unk1 or unk2:
        calls[site + " [has-unknowable-in-scope]"] += 1
    if r1 != orig or r2 != orig:
        flips[site] += 1
        _s, chain, event, ints = _context()
        rec = {"kind": "has_unfilled", "site": site, "orig": orig, "r1": r1, "r2": r2,
               "window": [int(start_idx), int(end_idx)], "check_to_idx": int(check_to_idx),
               "check_to_time": _time_at(df, int(check_to_idx)), "direction": direction,
               "unknowable_r1": unk1, "unknowable_r2": unk2, "entity": _entity(df),
               "event": event, "ints": ints, "chain": chain[:10]}
        _fh.write(json.dumps(rec, default=str) + "\n")
    return orig


def _make_variant(repl):
    src = textwrap.dedent(inspect.getsource(_orig_sweep))
    needle = "enter_idx = max(inst.start_idx, first_active)"
    assert src.count(needle) == 1, "sweep enter line not found"
    src = src.replace(needle, repl)
    ns = dict(vars(poi_mod))
    exec(compile(src, "<sweep-variant>", "exec"), ns)
    return ns["_compute_poi_activation_history"]


_sweep_r1 = _make_variant("enter_idx = max(inst.start_idx + 1, first_active)")
_sweep_r2 = _make_variant("enter_idx = max(inst.end_idx + 1, first_active)")


def _hist_key(h):
    return [(e["idx"], e["active"], tuple(e.get("versions", []))) for e in h]


def shadow_sweep(df, **kw):
    orig = _orig_sweep(df, **kw)
    r1 = _sweep_r1(df, **kw)
    r2 = _sweep_r2(df, **kw)
    calls["POI_SWEEP"] += 1
    if _hist_key(r1) != _hist_key(orig) or _hist_key(r2) != _hist_key(orig):
        flips["POI_SWEEP"] += 1
        fr_locals = {}
        for fr in _frames(1, 4):
            if fr.f_code.co_name == "derive_poi_zones":
                loc = fr.f_locals
                fr_locals = {"sid": loc.get("sid"), "cycle_id": loc.get("cycle_id"),
                             "lifecycle_floor": loc.get("lifecycle_floor")}
        rec = {"kind": "poi_sweep", "ic_idx": kw["ic_idx"], "sd": kw["sd"],
               "cts_established_idx": kw["cts_established_idx"], "scan_end": kw["scan_end"],
               "floor": kw.get("lifecycle_floor_idx"), "ctx": fr_locals,
               "entity": _entity(df),
               "orig": _hist_key(orig), "r1": _hist_key(r1), "r2": _hist_key(r2)}
        _fh.write(json.dumps(rec, default=str) + "\n")
    return orig


for mod in (imb_mod, ms_mod, fib_mod, poi_mod, ccf_mod):
    if hasattr(mod, "has_unfilled_imbalance"):
        mod.has_unfilled_imbalance = shadow_has
poi_mod._compute_poi_activation_history = shadow_sweep

# `get_unfilled_imbalances` was deleted 2026-09-30 (no production caller — the
# 0 GET_UNFILLED calls this script measured for Plan F); shadow it only if present.
_orig_get = getattr(imb_mod, "get_unfilled_imbalances", None)


def shadow_get(*a, **k):
    calls["GET_UNFILLED " + _site_only()] += 1
    return _orig_get(*a, **k)


if _orig_get is not None:
    imb_mod.get_unfilled_imbalances = shadow_get
    fib_mod.get_unfilled_imbalances = shadow_get


@atexit.register
def _summary():
    _fh.close()
    print("\n=== SHADOW SUMMARY (calls / flips per site) ===")
    for site in sorted(calls):
        print(f"{calls[site]:7d} calls  {flips.get(site, 0):5d} flips  {site}")


sys.argv = ["engine_v2.run_replay"]
runpy.run_module("engine_v2.run_replay", run_name="__main__", alter_sys=True)
