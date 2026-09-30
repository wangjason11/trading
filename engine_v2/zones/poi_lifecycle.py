"""POI activation-history primitives (pure, leaf module).

A POI's `meta["activation_history"]` is a list of `{"idx", "active", ...}`
events produced by `poi_zones._compute_poi_activation_history`. A POI can
flap active -> inactive -> active multiple times within a cycle, so its live
state is NOT representable by the single scalar `meta["confirmed_idx"]`
(which collapses to the LAST activate idx). Any consumer that needs "is the
POI active AS OF candle `idx`?" must walk the history per-candle.

This module is the single source of that walk. It is imported by:
  - `zones/zone_proximity.py` (the sd:POI proximity gate)
  - `charting/_zone_render.py` (POI fill stretches)
  - the chart hover labels (`export_plotly`, `export_m15_chart`)

It deliberately imports nothing from `zones/` or `charting/` so it stays a
leaf with no cycle risk.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


def active_stretches_from_history(
    history: Sequence[Dict[str, Any]],
    open_end_idx: int,
) -> List[Tuple[int, int]]:
    """Pair each `active=True` event with the next `active=False` event into
    `(start_idx, end_idx)` stretches (both ends inclusive).

    - A trailing activate with no following deactivation yields a stretch
      ending at `open_end_idx`.
    - Events with `idx > open_end_idx` are truncated (any open stretch is
      closed at `open_end_idx` and the walk stops).
    - A deactivation at idx D closes the open stretch at D-1 (the last
      active candle), guarded against a degenerate same-idx flip.

    Pure; the caller supplies the history list and the open-ended cap.
    """
    stretches: List[Tuple[int, int]] = []
    active_start: Optional[int] = None
    for ev in history:
        idx = int(ev["idx"])
        if idx > open_end_idx:
            if active_start is not None:
                stretches.append((active_start, open_end_idx))
                active_start = None
            return stretches
        if ev.get("active"):
            if active_start is None:
                active_start = idx
        else:
            if active_start is not None:
                end_idx = max(active_start, idx - 1)
                stretches.append((active_start, end_idx))
                active_start = None
    if active_start is not None:
        stretches.append((active_start, open_end_idx))
    return stretches


def poi_active_as_of(zone: Any, idx: int) -> bool:
    """True iff the POI is active AT candle `idx`, per its activation history.

    Respects `meta["end_idx"]` as a hard lifecycle cap (once ended, never
    active). Within the history, `idx` is active iff it falls inside an
    active stretch.
    """
    end_idx = zone.meta.get("end_idx")
    if end_idx is not None and idx > int(end_idx):
        return False
    history = zone.meta.get("activation_history", []) or []
    stretches = active_stretches_from_history(history, open_end_idx=idx)
    return any(s <= idx <= e for s, e in stretches)


def poi_confirmed_idx_as_of(zone: Any, idx: int) -> Optional[int]:
    """The start idx of the active stretch containing `idx` (the "confirmed
    idx as of `idx`"), or None when the POI is inactive at `idx`.

    This is the per-candle replacement for the lossy scalar
    `meta["confirmed_idx"]` — e.g. for a POI that activated at 905, the value
    at candle 926 is 905, not the final-collapsed 997.
    """
    end_idx = zone.meta.get("end_idx")
    if end_idx is not None and idx > int(end_idx):
        return None
    history = zone.meta.get("activation_history", []) or []
    stretches = active_stretches_from_history(history, open_end_idx=idx)
    for s, e in stretches:
        if s <= idx <= e:
            return s
    return None
