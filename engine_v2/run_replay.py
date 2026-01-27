from __future__ import annotations

from pathlib import Path
from datetime import timedelta
from typing import Optional, Tuple
from engine_v2.config import CONFIG
from engine_v2.data.provider_oanda import get_history
from engine_v2.pipeline.orchestrator import run_pipeline
from engine_v2.charting.export_plotly import export_chart_plotly
from engine_v2.structure.identify_start import identify_start_scenario_1

chart_cfg = {
    # Week 4 “pattern work mode” defaults:
    "candle_types": {"pinbar": False, "maru": False},  
    "patterns": {"engulfing": False, "star": False, "continuous": True, "double_maru": True, "one_maru_continuous": True, "one_maru_opposite": True},

    "struct_state": {"labels": True},
    "range_visual": {"rectangles": True},

    # keep these off for now (declutter)
    "structure": {"levels": True, "swings": False},
    "zones": {"KL": True, "OB": False},
}


def export_charts_with_optional_zoom(
    *,
    df,
    structure_levels,
    chart_cfg,
    title_prefix: str,
    basename: str,
    zoom: Optional[Tuple[int, int]] = None,
):
    """
    Always exports a full chart.
    Optionally exports a zoomed chart if `zoom=(i0, i1)` is provided.
    """

    # --- Full chart ---
    paths_full = export_chart_plotly(
        df,
        title=f"{title_prefix} (full)",
        basename=basename,
        structure_levels=structure_levels,
        cfg=chart_cfg,
    )

    print(f"Chart FULL HTML: {paths_full.html_path}")
    print(f"Chart FULL PNG : {paths_full.png_path}")

    # --- Optional zoom chart ---
    if zoom is not None:
        i0, i1 = zoom
        zoom_basename = f"{basename}_zoom_{i0}-{i1}"

        paths_zoom = export_chart_plotly(
            df,
            title=f"{title_prefix} (zoom {i0}–{i1})",
            basename=zoom_basename,
            structure_levels=structure_levels,
            cfg=chart_cfg,
            idx_range=(i0, i1),
        )

        print(f"Chart ZOOM HTML: {paths_zoom.html_path}")
        print(f"Chart ZOOM PNG : {paths_zoom.png_path}")

    return paths_full

def _fmt_float(x: float) -> str:
    # filename-safe float: 0.0001 -> "0p0001"
    s = f"{x:.10f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")

def make_basename(*, pair: str, timeframe: str, start, end, struct_direction: int, eps: float, range_min_k: int, range_max_k: int) -> str:
    return (
        f"{pair}_{timeframe}_{start.date()}_{end.date()}"
        f"_sd{struct_direction}_eps{_fmt_float(eps)}_rk{range_min_k}-{range_max_k}"
    )

def export_csv(df, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def _timeframe_to_timedelta(tf: str) -> timedelta:
    tf = str(tf).upper().strip()
    if tf.startswith("S"):
        return timedelta(seconds=int(tf[1:]))
    if tf.startswith("M"):
        return timedelta(minutes=int(tf[1:]))
    if tf.startswith("H"):
        return timedelta(hours=int(tf[1:]))
    if tf in ("D", "DAILY"):
        return timedelta(days=1)
    raise ValueError(f"Unsupported timeframe for auto-extend: {tf}")


def _floor_to_day_start(ts):
    return ts.replace(hour=0, minute=0, second=0, microsecond=0)

def fetch_history_with_auto_extend(
    *,
    pair: str,
    timeframe: str,
    start,
    end,
    lookback_days: int = 183,
    min_history: int = 50,
    max_extend_iters: int = 8,
    extend_days: int = 4,
):
    """
    Week 6 Part 3A requirement:
    If identify_start_scenario_1 indicates the chosen extreme is 'too early',
    extend the dataset earlier and retry until it passes (or guard trips).
    """
    cur_start = _floor_to_day_start(start)
    last_decision = None
    for k in range(int(max_extend_iters)):
        df = get_history(pair=pair, timeframe=timeframe, start=cur_start, end=end)

        input_idx = int(df.index.max())
        decision = identify_start_scenario_1(
            df, input_idx=input_idx, lookback_days=lookback_days, min_history=min_history
        )
        last_decision = decision

        if not decision.meta.get("too_early", False):
            return df, cur_start, decision

        raw_start_idx = int(decision.meta.get("raw_start_idx", decision.start_idx))

        prev_start = cur_start
        cur_start = _floor_to_day_start(cur_start - timedelta(days=int(extend_days)))

        print(
            f"[auto_extend] iter={k+1} too_early raw_start_idx={raw_start_idx} "
            f"min_history={min_history} => extending start by {extend_days} days "
            f"{prev_start} -> {cur_start}"
        )

    print("[auto_extend] WARNING: hit max_extend_iters; proceeding with last fetched dataset.")
    df = get_history(pair=pair, timeframe=timeframe, start=cur_start, end=end)

    input_idx = int(df.index.max())
    last_decision = identify_start_scenario_1(
        df, input_idx=input_idx, lookback_days=lookback_days, min_history=min_history
    )
    return df, cur_start, last_decision


def main() -> None:
    # df = get_history(
    #     pair=CONFIG.pair,
    #     timeframe=CONFIG.timeframe,
    #     start=CONFIG.start,
    #     end=CONFIG.end,
    # )

    df, effective_start, start_decision = fetch_history_with_auto_extend(
        pair=CONFIG.pair,
        timeframe=CONFIG.timeframe,
        start=_floor_to_day_start(CONFIG.start),
        end=CONFIG.end,
        lookback_days=183,
        min_history=50,
        extend_days=4,
        max_extend_iters=8,
    )
    df.attrs["pair"] = CONFIG.pair

    raw_path = (
        f"artifacts/debug/"
        f"{CONFIG.pair}_{CONFIG.timeframe}_{effective_start.date()}_{CONFIG.end.date()}_raw.csv"
    )
    export_csv(df.copy(), raw_path)

    res = run_pipeline(df)

    print(res.df["is_range"].value_counts())
    print(res.df[res.df["is_range"] == 1][["is_range_confirm_idx", "is_range_lag"]].head())

    from engine_v2.debug.export_structure import export_levels, export_swings
    from engine_v2.structure.structure_v1 import compute_structure_levels

    # Use the same parameters you settled on for the "good range"
    swings, levels = compute_structure_levels(res.df, left=6, right=6)

    print("Exported:", "artifacts/debug/swings.csv", "artifacts/debug/structure_levels.csv")

    res.df.attrs["structure_levels"] = res.structure
    res.df.attrs["kl_zones"] = res.meta.get("kl_zones", [])

    print("=== Replay Summary ===")
    print(f"pair={CONFIG.pair} tf={CONFIG.timeframe}")
    print(f"candles={len(res.df)}")
    print(f"pattern_events={len(res.patterns)}")
    print(f"structure_levels={len(res.structure)}")
    print("notes:", res.meta.get("notes", {}))

    final_path = (
        f"artifacts/debug/"
        f"{CONFIG.pair}_{CONFIG.timeframe}_{effective_start.date()}_{CONFIG.end.date()}_final.csv"
    )
    export_csv(res.df, final_path)

    print(f"Raw data: {raw_path}")
    print(f"Full pipeline data: {final_path}")

    # Export chart artifacts
    # basename = f"{CONFIG.pair}_{CONFIG.timeframe}_{CONFIG.start.date()}_{CONFIG.end.date()}"

    # Keep in sync with structure_engine MarketStructure(...) args for now
    # struct_direction = 1

    sd_series = res.df["struct_direction"].astype(int)
    sd_last = int(sd_series[sd_series != 0].iloc[-1]) if (sd_series != 0).any() else 0
    struct_direction = sd_last

    eps = 0.0001
    range_min_k = 2
    range_max_k = 5

    basename = make_basename(
        pair=CONFIG.pair,
        timeframe=CONFIG.timeframe,
        start=effective_start,
        end=CONFIG.end,
        struct_direction=struct_direction,
        eps=eps,
        range_min_k=range_min_k,
        range_max_k=range_max_k,
    )
    
    print("DEBUG structure_levels:", len(res.structure))

    export_swings(swings, f"artifacts/debug/{basename}_swings.csv")
    export_levels(levels, f"artifacts/debug/{basename}_structure_levels.csv")

    # ---------------------------
    # Chart export
    # ---------------------------

    TITLE_PREFIX = f"{CONFIG.pair} {CONFIG.timeframe}"
    ZOOM = None              # e.g. None or (300, 520)

    export_charts_with_optional_zoom(
        df=res.df,
        structure_levels=res.structure,
        chart_cfg=chart_cfg,
        title_prefix=TITLE_PREFIX,
        basename=basename,
        zoom=ZOOM,
    )


    from engine_v2.debug.export_zones import export_kl_zones
    export_kl_zones(res.meta.get("kl_zones", []), f"artifacts/debug/{basename}_kl_zones.csv")


if __name__ == "__main__":
    main()
