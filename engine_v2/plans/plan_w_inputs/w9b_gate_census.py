"""PLAN_W 9b census — every fib record the POI record gate drops, on any window.

The gate (`poi_zones._poi_record_gate` + the CTS_ESTABLISHED check in `select_ic_records`; FIB_LIFECYCLE_SPEC §11): a
fib record gets a POI record iff (active AND its cycle not ended) OR locked — AND its cycle has a CTS_ESTABLISHED.
This script wraps `orchestrator._run_downstream_pipeline` (the H1 main and every sub projection run it) and lists, per
run, each record that is the cycle's LIVE version (`active`, the condition axis) yet gets no POI record:
  - ENDED_UNLOCKED — its cycle ended before the fib locked (the 9b case);
  - NO_CTS_EST     — it passes the gate but its cycle has no CTS_ESTABLISHED (option N, 2026-09-29).
Superseded cross versions (`active` False) and condition-inactive records are not listed (no POI either way).
Indices are the run's frame (a sub's is slice-local) — the times are printed beside them.

    PYTHONPATH=. python engine_v2/plans/plan_w_inputs/w9b_gate_census.py START END OUT.txt   (from the repo root;
    START / END as YYYY-MM-DD, UTC; the replay's fetch, floor seeds and lower TFs)
"""
import dataclasses
import sys
from datetime import datetime, timezone

import engine_v2.run_replay as rr
from engine_v2.pipeline import orchestrator
from engine_v2.zones.poi_zones import _poi_record_gate

start, end, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
rr.CONFIG = dataclasses.replace(
    rr.CONFIG,
    start=datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc),
    end=datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc),
)

found = {}       # dedupe key -> line
runs = []
_orig = orchestrator._run_downstream_pipeline


def _t(df, idx):
    try:
        return str(df["time"].iloc[int(idx)])[:16]
    except Exception:
        return "?"


def wrapped(df, events, struct_direction, **kw):
    res = _orig(df, events, struct_direction, **kw)
    path = kw.get("structure_path_id", "H1.main")
    pfx = kw.get("log_prefix", "")
    est = {}
    conf = {}
    for ev in events:
        k = (int(ev.meta.get("structure_id", 0)), int(ev.meta.get("cycle_id", 0)))
        if ev.type == "CTS_ESTABLISHED":
            est[k] = int(ev.meta["confirmed_at"])
        elif ev.type == "CTS_CONFIRMED":
            conf.setdefault(k, int(ev.idx))
    n_records = 0
    for fs in res["fib_tracker"].get_fibs_for_charting():
        n_records += 1
        k = (int(fs.structure_id), int(fs.cycle_id))
        if not fs.active or fs.locked:
            continue
        if fs.end_idx is not None:
            kind = "ENDED_UNLOCKED"
        elif k not in est:
            kind = "NO_CTS_EST"
        else:
            continue
        assert kind == "NO_CTS_EST" or not _poi_record_gate(fs)
        key = (path, pfx, kind, k, int(fs.bos_idx), int(fs.cts_idx), fs.end_idx)
        e = est.get(k)
        found[key] = (
            f"{kind:<15} {path} {pfx} sid {k[0]} cyc {k[1]} v{fs.meta.get('version', '?')} "
            f"dir {fs.struct_direction:+d} | fib bos {fs.bos_idx} ({_t(df, fs.bos_idx)}) -> cts {fs.cts_idx} "
            f"({_t(df, fs.cts_idx)}) | live {fs.start_idx} ({_t(df, fs.start_idx) if fs.start_idx is not None else '-'})"
            f" -> {fs.end_idx} ({_t(df, fs.end_idx)}) end_reason={fs.end_reason} | CTS_ESTABLISHED moment "
            f"{e} ({_t(df, e) if e is not None else '-'}), CTS_CONFIRMED {conf.get(k)} | frame n={len(df)} "
            f"cap={kw.get('lifecycle_cap')} cap_reason={kw.get('cap_reason')}")
    runs.append((path, pfx, n_records))
    return res


orchestrator._run_downstream_pipeline = wrapped

df, effective_start, _ = rr.fetch_history_with_auto_extend(
    pair=rr.CONFIG.pair, timeframe=rr.CONFIG.timeframe, start=rr._floor_to_day_start(rr.CONFIG.start),
    end=rr.CONFIG.end, lookback_days=183, min_history=50, extend_days=4, max_extend_iters=8,
)
df.attrs["pair"] = rr.CONFIG.pair
seeds = rr.fetch_floor_seeds(rr.CONFIG.pair, rr.CONFIG.timeframe, df["time"].iloc[0], rr.CONFIG.lower_timeframes)
res = rr.run_pipeline(df, lower_timeframes=rr.CONFIG.lower_timeframes, floor_seeds=seeds)

lines = [f"window {rr.CONFIG.pair} {rr.CONFIG.timeframe} {start} -> {end} (fetched from {effective_start}); H1 frame "
         f"{len(df)} candles; {len(runs)} downstream runs, {sum(r[2] for r in runs)} fib records",
         f"dropped live records: {len(found)} "
         f"(ENDED_UNLOCKED {sum(1 for k in found if k[2] == 'ENDED_UNLOCKED')}, "
         f"NO_CTS_EST {sum(1 for k in found if k[2] == 'NO_CTS_EST')})", ""]
lines += [found[k] for k in sorted(found, key=str)]
text = "\n".join(lines) + "\n"
print(text)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(text)
