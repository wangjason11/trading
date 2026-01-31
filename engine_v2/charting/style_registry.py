"""
Style Registry for Chart Rendering.

IMPORTANT: ALL chart styling and formatting parameters should be defined here,
not hardcoded in export_plotly.py. This includes:
- Colors, line widths, opacity values
- Marker sizes and symbols
- Fill colors and border styles
- Opacity tier multipliers

When adding new chart elements, always add their styles here first,
then reference them via _style() in export_plotly.py.
"""
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
        "fillcolor": "gray",
        "opacity": 0.1,
        "line_width": 0,
        "layer": "below",
    },

    # Week 5: confirmed structure swing line + points
    "structure.swing_line": {
        "line": {"width": 2, "color": "black"},
        "opacity": 0.9,
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
        "fill_opacity_active": 0.4,
        "fill_opacity_inactive": 0.15,
        "confirm_line_width": 2,
        "confirm_opacity_active": 0.9,
        "confirm_opacity_inactive": 0.30,
    },
    "zone.kl.sell": {
        "rgb": "220, 0, 0",
        "fill_opacity_active": 0.4,
        "fill_opacity_inactive": 0.15,
        "confirm_line_width": 2,
        "confirm_opacity_active": 0.9,
        "confirm_opacity_inactive": 0.30,
    },

    # Transparent hover "hitbox" lines (shapes don't hover)
    "zone.kl.hover_line": {
        "line": {"width": 6, "color": "rgba(0,0,0,0)"},
        "showlegend": False,
    },

    # -----------------------------
    # Week 7: POI Zones (rectangles)
    # -----------------------------
    "zone.poi.buy": {
        "rgb": "0, 100, 180",
        "fill_opacity_active": 0.35,
        "fill_opacity_inactive": 0.12,
        "confirm_line_width": 2,
        "confirm_opacity_active": 0.85,
        "confirm_opacity_inactive": 0.25,
    },
    "zone.poi.sell": {
        "rgb": "180, 100, 0",
        "fill_opacity_active": 0.35,
        "fill_opacity_inactive": 0.12,
        "confirm_line_width": 2,
        "confirm_opacity_active": 0.85,
        "confirm_opacity_inactive": 0.25,
    },
    "zone.poi.hover_line": {
        "line": {"width": 6, "color": "rgba(0,0,0,0)"},
        "showlegend": False,
    },

    # -----------------------------
    # Week 7: Fibonacci levels
    # -----------------------------
    # Dotted lines at 0% and 100% anchors
    "fib.anchor_line.active": {
        "line": {"width": 1.5, "dash": "dot", "color": "rgba(0, 0, 139, 0.8)"},  # dark blue
    },
    "fib.anchor_line.historical": {
        "line": {"width": 1.5, "dash": "dot", "color": "rgba(0, 0, 139, 0.4)"},  # dark blue, faded
    },
    # Rectangle between 61.8% and 80%
    "fib.zone_rect.active": {
        "line": {"width": 1.5, "dash": "dot", "color": "rgba(0, 0, 139, 0.8)"},  # dark blue border
        "fillcolor": "rgba(0, 0, 139, 0.0)",  # transparent fill
    },
    "fib.zone_rect.historical": {
        "line": {"width": 1.5, "dash": "dot", "color": "rgba(0, 0, 139, 0.4)"},  # dark blue border, faded
        "fillcolor": "rgba(0, 0, 139, 0.0)",  # transparent fill
    },
    "fib.label": {
        "font": {"size": 9, "color": "gray"},
    },

    # -----------------------------
    # Week 7: Imbalance candle highlighting
    # -----------------------------
    "imbalance.bullish": {
        # Lime Green at 80% opacity - used for entire candle (body + wicks)
        "rgba": "rgba(50, 205, 50, 0.8)",
    },
    "imbalance.bearish": {
        # Gold at 80% opacity - used for entire candle (body + wicks)
        "rgba": "rgba(255, 215, 0, 0.8)",
    },

    # -----------------------------
    # Hover lines (invisible lines for hover on shapes)
    # -----------------------------
    "hover_line.range": {
        "line": {"width": 2, "color": "rgba(0,0,0,0)"},
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

    # -----------------------------
    # Week 7: Prev BOS Line (after reversal)
    # -----------------------------
    "prev_bos_line": {
        "line": {"width": 2, "color": "black"},
    },

    # -----------------------------
    # Global: Opacity tiers for multi-structure visibility
    # -----------------------------
    "opacity_tiers": {
        "active": 1.0,           # Active elements (current cycle, not locked)
        "recent_inactive": 0.5,  # Inactive elements from most recent structure
        "prior_inactive": 0.2,   # Inactive elements from prior structures
    },
}
