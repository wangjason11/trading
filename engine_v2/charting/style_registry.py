from __future__ import annotations

STYLE = {
    # candle types
    "candle.pinbar": {
        "marker": {
            "size": 6,
            "opacity": 0.9,
            "symbol": "circle",
        }
    },
    "candle.maru": {
        "marker": {
            "size": 6,
            "opacity": 0.9,
            "symbol": "square",
        }
    },

    # patterns
    "pattern.engulfing.up": {"marker": {"size": 9}},
    "pattern.engulfing.down": {"marker": {"size": 9}},
    "pattern.star.up": {"marker": {"size": 9}},
    "pattern.star.down": {"marker": {"size": 9}},

    # "structure_pattern.up": {"marker": {"size": 9}},
    # "structure_pattern.down": {"marker": {"size": 9}},

    # Week 4 structure breakout/pullback markers
    "structure.success.up": {
        "marker": {"size": 9, "symbol": "triangle-up", "opacity": 0.9, "color": "green"}
    },
    "structure.success.down": {
        "marker": {"size": 9, "symbol": "triangle-down", "opacity": 0.9, "color": "red"}
    },
    "structure.confirmed.up": {
        "marker": {"size": 9, "symbol": "triangle-up", "opacity": 0.5, "color": "green"}
    },
    "structure.confirmed.down": {
        "marker": {"size": 9, "symbol": "triangle-down", "opacity": 0.5, "color": "red"}
    },

    # Week 4 range candle markers
    "range.candle": {
        "marker": {"size": 3, "symbol": "circle", "opacity": 0.9, "color": "orange"}
    },

    # Week 5: range rectangles (shape styling)
    "range.rect": {
        "fillcolor": "yellow",
        "opacity": 0.20,
        "line_width": 0,
        "layer": "below",
    },

    # Week 5: confirmed structure swing line + points
    "structure.swing_line": {
        "line": {"width": 2, "color": "black"},
    },
    "structure.cts": {
        "marker": {"size": 3, "symbol": "circle", "opacity": 0.95, "color": "black"},
    },
    "structure.bos": {
        "marker": {"size": 3, "symbol": "circle", "opacity": 0.95, "color": "black"},
    },

    # Week 5: reversal watch visuals
    "structure.reversal_watch_line": {
        "line": {"width": 2, "dash": "dot", "color": "purple"},
    },
    "structure.reversal_watch_start": {
        "marker": {"size": 8, "symbol": "x", "opacity": 0.95, "color": "purple"},
    },

    # -----------------------------
    # Week 6: KL Zones (rectangles)
    # -----------------------------
    "zone.kl.buy": {
        "rgb": "0, 180, 0",
        "fill_opacity_active": 0.35,
        "fill_opacity_inactive": 0.15,
        "confirm_line_width": 2,
        "confirm_opacity_active": 0.70,
        "confirm_opacity_inactive": 0.30,
    },
    "zone.kl.sell": {
        "rgb": "220, 0, 0",
        "fill_opacity_active": 0.35,
        "fill_opacity_inactive": 0.15,
        "confirm_line_width": 2,
        "confirm_opacity_active": 0.70,
        "confirm_opacity_inactive": 0.30,
    },

    # Transparent hover “hitbox” lines (shapes don't hover)
    "zone.kl.hover_line": {
        "line": {"width": 6, "color": "rgba(0,0,0,0)"},
        "showlegend": False,
    },

    # -----------------------------
    # Global chart layout defaults
    # -----------------------------
    "chart.layout": {
        # white (or set to "rgba(0,0,0,0)" for transparent)
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
    },

    "chart.axis": {
        # grid
        "showgrid": True,
        "gridcolor": "rgba(0,0,0,0.12)",
        "griddash": "dot",
        "gridwidth": 1,
        "zeroline": False,

        # axis box / border
        "showline": True,
        "linecolor": "rgba(0,0,0,0.4)",   # solid dark grey
        "linewidth": 1,
        "mirror": True,                  # draw full rectangle (top/right included)
    },
}
