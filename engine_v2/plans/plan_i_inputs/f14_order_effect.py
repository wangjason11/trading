"""PLAN_I §9.2 F16 / F17 (session 4, 2026-10-08) — input to the S5 round (R8-1).

Does the event ORDER change the downstream fib? One search trial, fed to FibTracker twice: in
`processing_order_key` order (today's downstream) and in MS's emission order (S5's at-emit tracker).

  python f14_order_effect.py real|waves SEED T

Prints every fib record (key, active / locked, BOS / CTS anchors, meta) per order and whether they differ, for both
modes (h1, cross_cycle). Dispatch = the orchestrator loop (BOS per cycle collected up front so emission order never
lacks it)."""
import dataclasses, importlib.util, json, sys

import engine_v2.structure.event_fields as ef
from engine_v2.zones.fib_tracker import FibTracker, FibTrackerConfig

spec = importlib.util.spec_from_file_location(
    "rts", "engine_v2/plans/plan_e_inputs/review_scripts/random_tail_search.py")
rts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rts)


def build(stream, seed, t_want):
    gen = rts.real_trials(int(seed)) if stream == "real" else rts.wave_trials(int(seed))
    for tr in gen:
        if tr[0] == int(t_want):
            break
    if stream == "real":
        t, sd, tf, off, start, length, jit, rows = tr
    else:
        t, sd, tf, base, rows, nw = tr
        start = 0
    df = rts._prepare_df(rows)
    ms = rts._make_market_structure(df, struct_direction=sd, start_idx=start, structure_id=0, timeframe=tf,
                                    pip_size=rts._pip_size_from_pair(df))
    ms.run()
    return df, ms.events


def feed(df, events, mode):
    """`mode` picks the LIMITS (one FibTracker logic since PLAN_I S4): "main" = production's main limits (sid 0: no P →
    singles), "sub" = `unreached_for` (never beyond → every cross the walk allows)."""
    from engine_v2.tests._cross_fib_limits import main_limits, unreached_for
    trk = FibTracker(FibTrackerConfig(fib_levels=[30.0, 50.0, 61.8, 80.0], fill_threshold=0.70),
                     cross_fib_limits=main_limits(events) if mode == "main" else unreached_for(events))
    bos = {}
    for e in events:
        if e.type == "BOS_CONFIRMED":
            bos[(int(e.meta["structure_id"]), int(e.meta["cycle_id"]))] = (ef.bos_anchor_idx(e), e.price)
    log = []
    for e in events:
        k = (int(e.meta.get("structure_id", 0)), int(e.meta.get("cycle_id", 0)))
        try:
            if e.type == "CTS_ESTABLISHED" and k in bos:
                trk.on_cts_established(e, df, bos[k][0], bos[k][1])
            elif e.type == "CTS_UPDATED":
                trk.on_cts_updated(e, df)
            elif e.type == "CTS_CONFIRMED":
                trk.on_cts_confirmed(e)
            elif e.type == "CTS_THRESHOLD_UPDATED":
                trk.on_cts_threshold_updated(e, df)
            else:
                continue
        except Exception as x:  # noqa: BLE001
            return None, f"{type(x).__name__}: {x}"
        log.append((e.type, k[1], ef.stamped_idx(e)))
    recs = {}
    for f in trk.get_all_fibs():
        d = dataclasses.asdict(f)
        recs[str((d["structure_id"], d["cycle_id"], d.get("meta", {}).get("version")))] = d
    return recs, log


if __name__ == "__main__":
    stream, seed, t = sys.argv[1:4]
    df, events = build(stream, seed, t)
    emission = list(events)
    processing = sorted(events, key=ef.processing_order_key)
    for mode in ("main", "sub"):
        a, la = feed(df, processing, mode)
        b, lb = feed(df, emission, mode)
        if isinstance(la, str) or isinstance(lb, str):
            print(mode, "ERROR processing:", la if isinstance(la, str) else "-", "| emission:",
                  lb if isinstance(lb, str) else "-")
            continue
        same = json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
        print(f"== {mode}: fib records {'EQUAL' if same else 'DIFFER'} ({len(a)} vs {len(b)})")
        if not same:
            for key in sorted(set(a) | set(b), key=str):
                if json.dumps(a.get(key), sort_keys=True, default=str) != json.dumps(b.get(key), sort_keys=True, default=str):
                    print("  processing:", key, json.dumps(a.get(key), default=str)[:600])
                    print("  emission:  ", key, json.dumps(b.get(key), default=str)[:600])
