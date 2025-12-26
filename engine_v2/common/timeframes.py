from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Timeframe:
    """Simple timeframe representation."""

    code: str
    seconds: int


TIMEFRAMES = {
    "M1": Timeframe("M1", 60),
    "M2": Timeframe("M2", 2 * 60),
    "M5": Timeframe("M5", 5 * 60),
    "M15": Timeframe("M15", 15 * 60),
    "M30": Timeframe("M30", 30 * 60),
    "H1": Timeframe("H1", 60 * 60),
    "H4": Timeframe("H4", 4 * 60 * 60),
    "D": Timeframe("D", 24 * 60 * 60),
}


def require_timeframe(code: str) -> Timeframe:
    if code not in TIMEFRAMES:
        raise ValueError(
            f"Unsupported timeframe '{code}'. Supported: {sorted(TIMEFRAMES)}"
        )
    return TIMEFRAMES[code]
