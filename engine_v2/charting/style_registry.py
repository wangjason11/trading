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

    "structure_pattern.up": {"marker": {"size": 9}},
    "structure_pattern.down": {"marker": {"size": 9}},

    # structure
    "structure.level": {"line": {"width": 1, "dash": "dot"}},
}
