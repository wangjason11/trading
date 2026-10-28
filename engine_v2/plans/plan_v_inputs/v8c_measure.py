"""PLAN_V 8c measurement — run the pipeline on the CONFIG window (with its floor seeds, as the replay does) with the
qualified-candle rule as today (`base`) or with the agreed direction (`variant`: `agreed_direction(direction, vol_dir)
== required_dir`, which admits the volume-only doji — direction 0, vol_dir = the required direction), and dump to JSON:
every doji of the H1 frame and the M15 frame, every wave candle (H1 main + each unique sub) and every WVMI record with
its per-LP rows.

    PYTHONPATH=. python engine_v2/plans/plan_v_inputs/v8c_measure.py base|variant OUT.json   (from the repo root;
    a script run by path has its own folder on sys.path, not the repo — without PYTHONPATH no `engine_v2`)

The variant patches `wave_candles._is_qualified` and the name `wvmi` imported from it (the only two homes).
"""
import dataclasses
import json
import sys

import engine_v2.run_replay as rr
from engine_v2.features.candles_v2 import agreed_direction
from engine_v2.zones import wave_candles, wvmi

mode, out = sys.argv[1], sys.argv[2]
assert mode in ("base", "variant")
if mode == "variant":
    def _is_qualified_agreed(df, idx, required_dir):
        if idx not in df.index:
            return False
        direction = int(df.loc[idx, "direction"])
        vol_dir = int(df.loc[idx, "vol_dir"]) if "vol_dir" in df.columns else 0
        return int(agreed_direction(direction, vol_dir)) == required_dir
    wave_candles._is_qualified = _is_qualified_agreed
    wvmi._is_qualified = _is_qualified_agreed

df, effective_start, _ = rr.fetch_history_with_auto_extend(
    pair=rr.CONFIG.pair, timeframe=rr.CONFIG.timeframe, start=rr._floor_to_day_start(rr.CONFIG.start),
    end=rr.CONFIG.end, lookback_days=183, min_history=50, extend_days=4, max_extend_iters=8,
)
df.attrs["pair"] = rr.CONFIG.pair
seeds = rr.fetch_floor_seeds(rr.CONFIG.pair, rr.CONFIG.timeframe, df["time"].iloc[0], rr.CONFIG.lower_timeframes)
res = rr.run_pipeline(df, lower_timeframes=rr.CONFIG.lower_timeframes, floor_seeds=seeds)
lens_dfs = res.meta["sub_lens_dfs"]


def dojis(frame):
    rows = frame[frame["direction"] == 0]
    return [{"idx": int(i), "time": str(r["time"]), "o": float(r["o"]), "h": float(r["h"]), "l": float(r["l"]),
             "c": float(r["c"]), "vol_dir": int(r["vol_dir"]), "volume": float(r["volume"]),
             "candle_type": str(r["candle_type"])} for i, r in rows.iterrows()]


def wave(w, owner):
    return {"owner": owner, "structure_id": int(w.structure_id), "cycle_id": int(w.cycle_id),
            "source_kind": str(w.source_kind), "zone_side": str(w.zone_side),
            "last": w.last_wave_candle_idx, "first": w.first_wave_candle_idx}


def rec(r, owner):
    d = dataclasses.asdict(r)
    d["owner"] = owner
    d.pop("meta", None)
    return d


m15 = next(iter(lens_dfs.values()))
dump = {"h1_dojis": dojis(res.df), "m15_dojis": dojis(m15), "waves": [], "wvmi": []}
dump["waves"] += [wave(w, "H1") for w in res.df.attrs.get("wave_candles", [])]
dump["wvmi"] += [rec(r, "H1") for r in res.df.attrs.get("wvmi", [])]
seen_w, seen_r = set(), set()
for lens, ldf in sorted(lens_dfs.items()):
    for w in ldf.attrs.get("wave_candles", []):
        sub = w.meta.get("sub_id")
        k = (sub, w.structure_id, w.cycle_id, w.source_kind)
        if sub is not None and k not in seen_w:
            seen_w.add(k)
            dump["waves"].append(wave(w, f"sub{sub}"))
    for r in ldf.attrs.get("wvmi", []):
        sub = (r.meta or {}).get("sub_id")
        k = (sub, r.bos_structure_id, r.bos_cycle_id)
        if sub is not None and k not in seen_r:
            seen_r.add(k)
            dump["wvmi"].append(rec(r, f"sub{sub}"))
with open(out, "w", encoding="utf-8") as f:
    json.dump(dump, f, indent=1, default=str)
if mode == "base":      # the candle tables' rows (the frames are the same in both modes)
    cols = ["time", "o", "h", "l", "c", "volume", "direction", "vol_dir", "candle_type"]
    res.df[cols].to_csv(out.replace(".json", "_h1_frame.csv"), index_label="idx")
    m15[cols].to_csv(out.replace(".json", "_m15_frame.csv"), index_label="idx")
print(f"[v8c] {mode}: {len(dump['h1_dojis'])} H1 dojis, {len(dump['m15_dojis'])} M15 dojis, {len(dump['waves'])} wave "
      f"results, {len(dump['wvmi'])} WVMI records -> {out}")
