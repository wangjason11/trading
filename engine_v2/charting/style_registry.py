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
        "marker": {"size": 6, "symbol": "circle", "opacity": 0.9, "color": "orange"}
    },

    # structure
    "structure.level": {"line": {"width": 1, "dash": "dot"}},
}
