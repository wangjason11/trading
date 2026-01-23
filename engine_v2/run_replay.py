from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
from engine_v2.config import CONFIG
from engine_v2.data.provider_oanda import get_history
from engine_v2.pipeline.orchestrator import run_pipeline
from engine_v2.charting.export_plotly import export_chart_plotly

chart_cfg = {
    # Week 4 “pattern work mode” defaults:
    "candle_types": {"pinbar": False, "maru": False},  
    "patterns": {"engulfing": False, "star": False, "continuous": True, "double_maru": True, "one_maru_continuous": True, "one_maru_opposite": True},

    "struct_state": {"labels": True},
    "range_visual": {"rectangles": True},

    # keep these off for now (declutter)
    "structure": {"levels": True, "swings": False},
    "zones": {"KL": False, "OB": False},
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


def main() -> None:
    df = get_history(
        pair=CONFIG.pair,
        timeframe=CONFIG.timeframe,
        start=CONFIG.start,
        end=CONFIG.end,
    )

    raw_path = (
        f"artifacts/debug/"
        f"{CONFIG.pair}_{CONFIG.timeframe}_{CONFIG.start.date()}_{CONFIG.end.date()}_raw.csv"
    )
    export_csv(df.copy(), raw_path)

    res = run_pipeline(df)

    print(res.df["is_range"].value_counts())
    print(res.df[res.df["is_range"] == 1][["is_range_confirm_idx", "is_range_lag"]].head())

    from engine_v2.debug.export_structure import export_levels, export_swings
    from engine_v2.structure.structure_v1 import compute_structure_levels

    # Use the same parameters you settled on for the "good range"
    swings, levels = compute_structure_levels(res.df, left=6, right=6)

    export_swings(swings, "artifacts/debug/swings.csv")
    export_levels(levels, "artifacts/debug/structure_levels.csv")

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
        f"{CONFIG.pair}_{CONFIG.timeframe}_{CONFIG.start.date()}_{CONFIG.end.date()}_final.csv"
    )
    export_csv(res.df, final_path)

    print(f"Raw data: {raw_path}")
    print(f"Full pipeline data: {final_path}")

    # Export chart artifacts
    # basename = f"{CONFIG.pair}_{CONFIG.timeframe}_{CONFIG.start.date()}_{CONFIG.end.date()}"

    # Keep in sync with structure_engine MarketStructure(...) args for now
    struct_direction = 1
    eps = 0.0001
    range_min_k = 2
    range_max_k = 5

    basename = make_basename(
        pair=CONFIG.pair,
        timeframe=CONFIG.timeframe,
        start=CONFIG.start,
        end=CONFIG.end,
        struct_direction=struct_direction,
        eps=eps,
        range_min_k=range_min_k,
        range_max_k=range_max_k,
    )
    
    print("DEBUG structure_levels:", len(res.structure))

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
    export_kl_zones(res.meta.get("kl_zones", []), "artifacts/debug/kl_zones.csv")


if __name__ == "__main__":
    main()
