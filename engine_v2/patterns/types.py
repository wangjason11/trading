from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class PatternStatus(str, Enum):
    NONE = "NONE"
    SUCCESS = "SUCCESS"
    FAIL_NEEDS_CONFIRM = "FAIL_NEEDS_CONFIRM"
    CONFIRMED = "CONFIRMED"


@dataclass(frozen=True)
class PatternEvent:
    name: str
    direction: int  # +1 / -1

    start_idx: int
    end_idx: int  # index where the base pattern completes (2-candle => idx+1, continuous => idx+2)

    status: PatternStatus

    confirmation_threshold: Optional[float] = None
    confirmation_idx: Optional[int] = None

    break_threshold_used: Optional[float] = None

    debug: dict[str, Any] = field(default_factory=dict)
