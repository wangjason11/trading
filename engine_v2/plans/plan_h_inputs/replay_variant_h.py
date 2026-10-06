"""PLAN_H §8.4 — the REAL replay through a variant tree (`variant_patch.py`) + the behaviour-neutral resolver log.

    cd WORKDIR && python REPO/engine_v2/plans/plan_h_inputs/replay_variant_h.py TREE A_POI_ZONES LOG.json [--nolog]

WORKDIR holds a copy of `oanda.cfg`; the replay writes WORKDIR/artifacts/ (the main checkout is never touched, so
the variants can run side by side). TREE/engine_v2 is put first on sys.path (asserted). Unless `--nolog`:
  * every `structure_engine.compute_poi_inners_for_cycle` call (the only name MS reads the resolver by — resolved
    at `_make_market_structure` call time) is logged: the caller (main / sub_geometry / probe_phase2, from the
    stack), sid / cycle / sd, the AS-OF anchors (the fib the resolver built — after the Scenario-2 choice), the
    variant's inners, A's inners for the same arguments (A_POI_ZONES loaded as a separate module), both call
    times, and the IP pick + `select_ip_ic` time (PR8);
  * every other `select_ip_ic` call (the drawn POI in V1 / V3, the debug CSV) is logged with its pick, so the
    final drawn pick can be joined to the resolver's as-of picks (PR12);
  * a `select_ip_ic` exception is logged BEFORE it propagates (the sub-geometry scope swallows ValueError /
    IndexError — PLAN_H §8.5 step 3).
Indices are the frame's own (slice-local on a sub); `*_t` = the candle times, the join key across frames.
"""
import atexit
import importlib.util
import json
import os
import runpy
import sys
import time

tree = os.path.abspath(sys.argv[1])
a_pz_path = os.path.abspath(sys.argv[2])
log_path = os.path.abspath(sys.argv[3])
nolog = "--nolog" in sys.argv
sys.path.insert(0, tree)
import engine_v2.zones.poi_zones as pz  # noqa: E402
import engine_v2.structure.structure_engine as se  # noqa: E402
assert os.path.normcase(pz.__file__).startswith(os.path.normcase(tree)), pz.__file__
assert os.path.normcase(se.__file__).startswith(os.path.normcase(tree)), se.__file__

LOG = []
_ctx = []


def _t(df, i):
    if i is None or i < 0 or i >= len(df):
        return None
    return str(df["time"].iloc[int(i)])


def _caller():
    f = sys._getframe(2)
    while f is not None:
        fn = f.f_code.co_filename.replace("\\", "/")
        if fn.endswith("structure/unified_probe.py"):
            return "probe_phase2"
        if fn.endswith("multitf/entity_df_mutation.py"):
            return "sub_geometry"
        f = f.f_back
    return "main"


if not nolog:
    spec = importlib.util.spec_from_file_location("poi_zones_A", a_pz_path)
    A = importlib.util.module_from_spec(spec)
    sys.modules["poi_zones_A"] = A  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(A)

    real_select = pz.select_ip_ic

    def select_wrapped(df, fib_state, config, **kw):
        rec = {
            "tf": df.attrs.get("timeframe"), "frame_len": len(df), "frame_first_t": _t(df, 0),
            "sid": int(fib_state.structure_id), "cycle": int(fib_state.cycle_id),
            "sd": int(fib_state.struct_direction),
            "bos_idx": int(fib_state.bos_idx), "cts_idx": int(fib_state.cts_idx),
            "bos_t": _t(df, fib_state.bos_idx), "cts_t": _t(df, fib_state.cts_idx),
            "active": bool(fib_state.active), "locked": bool(fib_state.locked),
            "classify_misses": bool(kw.get("classify_misses", False)),
        }
        t0 = time.perf_counter()
        try:
            sel = real_select(df, fib_state, config, **kw)
        except Exception as e:  # logged, then re-raised (behaviour-neutral)
            rec.update(kind="select_error", err=repr(e), caller=_caller())
            LOG.append(rec)
            raise
        rec["ip_ms"] = (time.perf_counter() - t0) * 1000.0
        p = sel.pick
        rec.update(
            pick_idx=None if p is None else int(p.ic_idx), pick_t=None if p is None else _t(df, p.ic_idx),
            rank=None if p is None else p.ip_rank,
            pattern_confirmed_idx=None if p is None else p.pattern_confirmed_idx,
            n_cands=len(sel.candidates), tiebreak=None if sel.tiebreak is None else sel.tiebreak[0],
        )
        if _ctx:
            _ctx[-1]["select"] = rec
        else:
            rec["kind"] = "csv" if rec["classify_misses"] else "drawn"
            LOG.append(rec)
        return sel

    pz.select_ip_ic = select_wrapped
    real_resolver = se.compute_poi_inners_for_cycle

    def resolver_wrapped(df, bos_idx, bos_price, cts_idx, cts_price, struct_direction, structure_id=0,
                         cycle_id=0, fill_threshold=0.70, c0_data=None, *, fill_horizon_idx, snapshot_horizon_idx):
        args = (df, bos_idx, bos_price, cts_idx, cts_price, struct_direction, structure_id, cycle_id,
                fill_threshold, c0_data)
        kw = dict(fill_horizon_idx=fill_horizon_idx, snapshot_horizon_idx=snapshot_horizon_idx)
        entry = {
            "kind": "resolver", "caller": _caller(), "tf": df.attrs.get("timeframe"), "frame_len": len(df),
            "frame_first_t": _t(df, 0), "sid": int(structure_id), "cycle": int(cycle_id),
            "sd": int(struct_direction), "ms_bos_idx": int(bos_idx), "ms_cts_idx": int(cts_idx),
            "ms_bos_t": _t(df, bos_idx), "ms_cts_t": _t(df, cts_idx), "c0": c0_data is not None,
            "fill_horizon_idx": int(fill_horizon_idx),
        }
        _ctx.append(entry)
        t0 = time.perf_counter()
        try:
            out = real_resolver(*args, **kw)
        finally:
            _ctx.pop()
        entry["ms"] = (time.perf_counter() - t0) * 1000.0
        entry["inners"] = [float(x) for x in out]
        t1 = time.perf_counter()
        entry["today"] = [float(x) for x in A.compute_poi_inners_for_cycle(*args, **kw)]
        entry["today_ms"] = (time.perf_counter() - t1) * 1000.0
        LOG.append(entry)
        return out

    se.compute_poi_inners_for_cycle = resolver_wrapped

    # The activation attribution (the variant's drawn POIs): per POI its history, the history with cond5 forced
    # true (A's sweep, a 0 % variant threshold — cond1 + cond3 from the same floor without R1) and the variant's
    # sweep without the R1 floor -> which condition keeps a POI inactive.
    real_act = pz._compute_poi_activation_history

    def act_wrapped(df, **k):
        h = real_act(df, **k)
        if "pattern_confirmed_idx" in k:
            k2 = dict(k)
            pci = k2.pop("pattern_confirmed_idx")
            no_r1 = real_act(df, **{**k, "pattern_confirmed_idx": None})
            k2["variant_thresholds"] = {"X": 0.0}
            no_c5 = A._compute_poi_activation_history(df, **k2)
            f = lambda hh: ",".join(f"{e['idx']}{'A' if e['active'] else 'D'}:{e.get('reason')}" for e in hh)
            LOG.append({"kind": "act", "tf": df.attrs.get("timeframe"), "frame_first_t": _t(df, 0),
                        "ic_idx": int(k["ic_idx"]), "ic_t": _t(df, k["ic_idx"]),
                        "cts_established_idx": int(k["cts_established_idx"]), "pattern_confirmed_idx": pci,
                        "floor": k.get("lifecycle_floor_idx"), "scan_end": int(k["scan_end"]),
                        "hist": f(h), "no_r1": f(no_r1), "no_cond5_no_r1": f(no_c5)})
        return h

    pz._compute_poi_activation_history = act_wrapped


@atexit.register
def _dump():
    if nolog:
        return
    with open(log_path, "w") as fh:
        json.dump(LOG, fh)
    print(f"[replay_variant_h] {len(LOG)} log records -> {log_path}")


os.makedirs("artifacts/debug", exist_ok=True)  # some exporters do not create it
os.makedirs("artifacts/charts", exist_ok=True)
sys.argv = ["run_replay"]
runpy.run_module("engine_v2.run_replay", run_name="__main__")
