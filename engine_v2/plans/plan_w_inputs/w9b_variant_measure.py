"""PLAN_W 9b measurement — what a fib that ENDED without ever locking would get if the POI record gate let it through.

Runs the pipeline on a window (the replay's fetch, floor seeds, lower TFs) in one of two modes and dumps the final
outputs to JSON (entity-absolute indices, as the CSVs have them):
  - `base`     — the gate as coded: (active AND its cycle not ended) OR locked;
  - `gate_any` — (active) OR locked: the cycle's live version passes whether or not its cycle ended (the CTS_ESTABLISHED
                 requirement — option N, 2026-09-29 — kept: `select_ic_records` still skips a cycle without one).
Dumped: every POI zone and every IC record outcome of the H1 main and each M15 lens, every structure event, KL zone,
sub and trigger (to show nothing else moves) — `w9b_variant_diff.py` compares two dumps.

    PYTHONPATH=. python engine_v2/plans/plan_w_inputs/w9b_variant_measure.py MODE START END OUT.json   (repo root)
"""
import dataclasses
import json
import sys
from datetime import datetime, timezone

import engine_v2.run_replay as rr
from engine_v2.zones import poi_zones

mode, start, end, out_path = sys.argv[1:5]
assert mode in ("base", "gate_any")
if mode == "gate_any":
    poi_zones._poi_record_gate = lambda fs: bool(fs.active or fs.locked)
rr.CONFIG = dataclasses.replace(
    rr.CONFIG,
    start=datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc),
    end=datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc),
)

df, effective_start, _ = rr.fetch_history_with_auto_extend(
    pair=rr.CONFIG.pair, timeframe=rr.CONFIG.timeframe, start=rr._floor_to_day_start(rr.CONFIG.start),
    end=rr.CONFIG.end, lookback_days=183, min_history=50, extend_days=4, max_extend_iters=8,
)
df.attrs["pair"] = rr.CONFIG.pair
seeds = rr.fetch_floor_seeds(rr.CONFIG.pair, rr.CONFIG.timeframe, df["time"].iloc[0], rr.CONFIG.lower_timeframes)
res = rr.run_pipeline(df, lower_timeframes=rr.CONFIG.lower_timeframes, floor_seeds=seeds)


def _j(v):
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _j(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_j(x) for x in v]
    return str(v)


def poi(z):
    return {"sub_id": z.meta.get("sub_id"), "structure_id": z.meta.get("structure_id"), "cycle_id": z.meta.get("cycle_id"),
            "ic_idx": int(z.ic_idx), "side": z.side, "top": float(z.top), "bottom": float(z.bottom),
            "start_time": str(z.start_time), "end_time": str(z.end_time), "meta": _j(z.meta)}


def ic(r):
    return _j(dataclasses.asdict(r))


def ev(e):
    return {"type": e.type, "idx": int(e.idx), "meta": _j(e.meta)}


def frame(d):
    a = d.attrs
    return {"poi_zones": [poi(z) for z in a.get("poi_zones", []) or []],
            "ic_candidates": [ic(r) for r in a.get("ic_candidates", []) or []],
            "kl_zones": [_j({"side": z.side, "top": z.top, "bottom": z.bottom, "kind": z.source_kind, "meta": z.meta})
                         for z in a.get("kl_zones", []) or []],
            "events": [ev(e) for e in (a.get("events") or a.get("structure_events") or [])],
            "sids": [_j(dataclasses.asdict(s)) if dataclasses.is_dataclass(s) else _j(s) for s in a.get("sids", []) or []],
            "time": [str(t) for t in d["time"]]}


reg = res.meta["registry"]
dump = {"mode": mode, "window": [start, end], "H1.main": frame(res.df)}
for path in ("H1.main >> M15.counter", "H1.main >> M15.confluence"):
    entry = reg.get(path)
    if entry is not None:
        dump[path] = frame(entry.df)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(dump, f)
print("wrote", out_path, {k: len(v["poi_zones"]) for k, v in dump.items() if isinstance(v, dict)})
