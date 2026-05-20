"""Shared zone-rendering helpers (Item 5, 2026-05-20).

Pure functions consumed by both `export_plotly.py` (main H1 chart) and
`export_m15_chart.py` (sub M15 chart). Computes:

  - Active stretches for KL / POI zones (Spec 3: fill only where the zone
    is active).
  - The stepped-polygon outline path tracing the outer contour of a KL
    zone's `bounds_steps` (so multi-step expansions render as ONE outline,
    not N separate rect outlines).
  - Per-TF tier multipliers for sub-chart zone rendering (Spec 2).

Callers convert (x, y) coordinate lists into Plotly traces; this module
deliberately knows nothing about Plotly so it stays unit-testable.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

from engine_v2.charting.style_registry import STYLE


# ---------------------------------------------------------------------------
# Active stretches
# ---------------------------------------------------------------------------

def compute_kl_active_stretches(
    zone: Any,
    render_end_idx: int,
) -> List[Tuple[int, int]]:
    """KL zone active stretches.

    KL zones have no per-candle deactivation logic today, so they have at
    most one active stretch: `[(confirmed_idx, render_end_idx)]` when
    confirmed, else `[]`. `render_end_idx` should be the caller's effective
    last-rendered idx (e.g. derived from `zone.end_time` via the df, or the
    last candle for still-open zones).

    `confirmed_idx == -1` (sentinel for unconfirmed) yields `[]`.
    """
    confirmed_idx = int(zone.meta.get("confirmed_idx", -1))
    if confirmed_idx < 0:
        return []
    if confirmed_idx > render_end_idx:
        return []
    return [(confirmed_idx, render_end_idx)]


def compute_poi_active_stretches(
    zone: Any,
    render_end_idx: int,
) -> List[Tuple[int, int]]:
    """POI zone active stretches from `meta["activation_history"]`.

    Walks the history list pairing each `active=True` event with the next
    `active=False` event. A trailing activate with no deactivation yields
    a stretch ending at `render_end_idx`. All idx are clamped to
    `render_end_idx` so cascade-truncated zones don't render past their cap.

    Returned tuples are `(start_idx, end_idx)` with `end_idx` the LAST
    active candle (inclusive on both ends). Callers translate to time
    coordinates at render time.
    """
    history = zone.meta.get("activation_history", []) or []
    stretches: List[Tuple[int, int]] = []
    active_start: Optional[int] = None
    for ev in history:
        idx = int(ev["idx"])
        if idx > render_end_idx:
            if active_start is not None:
                stretches.append((active_start, render_end_idx))
                active_start = None
            return stretches
        if ev.get("active"):
            if active_start is None:
                active_start = idx
        else:
            if active_start is not None:
                # Deactivation at idx: last active candle is idx - 1.
                # Guard against degenerate same-idx flip (shouldn't happen
                # but cheap to defend).
                end_idx = max(active_start, idx - 1)
                stretches.append((active_start, end_idx))
                active_start = None
    if active_start is not None:
        stretches.append((active_start, render_end_idx))
    return stretches


# ---------------------------------------------------------------------------
# Stepped polygon outline (KL zone with bounds_steps)
# ---------------------------------------------------------------------------

def build_stepped_outline_xy(
    bounds_steps: Sequence[dict],
    x_resolver: Callable[[int], Any],
    x_start: Any,
    x_end: Any,
    fallback_top: float,
    fallback_bottom: float,
) -> Tuple[list, list]:
    """Stepped-polygon coordinates for a KL zone's outer contour.

    Walks all `bounds_steps` left-to-right and traces a closed polygon:
      - top edge: each step's `top` from step's x_start to x_end, with
        vertical jumps at step boundaries when `top` changes
      - right edge: vertical drop from last step's top to last step's bottom
      - bottom edge (R→L): each step's `bottom` with vertical jumps
      - left edge: vertical rise back to first step's top
      - close: repeat first point

    Single-step zones degenerate to a plain rectangle.

    `x_resolver(idx)` returns the x-coordinate (time) for a candle idx.
    `x_start` / `x_end` are the zone's overall x-range; each step's
    effective x is clamped into `[x_start, x_end]`. Steps that fall entirely
    outside the zone's x-range are dropped.

    `fallback_top` / `fallback_bottom` are used when `bounds_steps` is empty
    or all steps fall outside the zone's x-range — degenerate to a single
    rect with those bounds and the full `[x_start, x_end]`.

    Returns `(xs, ys)` lists ready to feed to `go.Scatter(mode='lines')`.
    """
    # Build per-step effective (x0, x1, top, bottom) tuples, clamped to zone range.
    sorted_steps = sorted(
        (s for s in bounds_steps),
        key=lambda s: int(s.get("start_idx", -1)),
    )
    effective: List[Tuple[Any, Any, float, float]] = []
    for k, s in enumerate(sorted_steps):
        step_start_idx = int(s.get("start_idx", -1))
        step_x0 = x_resolver(step_start_idx)
        if step_x0 is None:
            continue
        if step_x0 < x_start:
            step_x0 = x_start
        if k + 1 < len(sorted_steps):
            nxt_x0 = x_resolver(int(sorted_steps[k + 1].get("start_idx", -1)))
            step_x1 = nxt_x0 if nxt_x0 is not None else x_end
        else:
            step_x1 = x_end
        if step_x1 > x_end:
            step_x1 = x_end
        if step_x1 <= step_x0:
            continue
        effective.append((
            step_x0,
            step_x1,
            float(s.get("top", fallback_top)),
            float(s.get("bottom", fallback_bottom)),
        ))

    if not effective:
        # Fallback to plain rectangle path
        effective = [(x_start, x_end, fallback_top, fallback_bottom)]

    xs: list = []
    ys: list = []

    # Top edge L→R: emit (step_x0, step.top) then (step_x1, step.top); at each
    # boundary insert vertical jump if top differs between steps.
    for k, (sx0, sx1, top, _bot) in enumerate(effective):
        if k == 0:
            xs.append(sx0); ys.append(top)
        else:
            prev_top = effective[k - 1][2]
            if top != prev_top:
                xs.append(sx0); ys.append(prev_top)
                xs.append(sx0); ys.append(top)
        xs.append(sx1); ys.append(top)

    # Right edge (drop)
    last_x = effective[-1][1]
    last_bot = effective[-1][3]
    xs.append(last_x); ys.append(last_bot)

    # Bottom edge R→L: walk steps reversed
    for k in range(len(effective) - 1, -1, -1):
        sx0, sx1, _top, bot = effective[k]
        if k == len(effective) - 1:
            xs.append(sx0); ys.append(bot)
        else:
            next_bot = effective[k + 1][3]
            if bot != next_bot:
                xs.append(sx1); ys.append(bot)
            xs.append(sx0); ys.append(bot)

    # Left edge (rise) — close back to first point
    first_top = effective[0][2]
    xs.append(effective[0][0]); ys.append(first_top)

    return xs, ys


# ---------------------------------------------------------------------------
# Per-TF tier (sub-chart only)
# ---------------------------------------------------------------------------

def select_subordinate_tf_tier(
    zone_tf: Optional[str],
    primary_sub_tf: str,
    smallest_sub_tf: Optional[str] = None,
) -> float:
    """Per-TF tier multiplier for a zone rendered on a subordinate chart.

    - zone's TF matches `smallest_sub_tf` (3-TF future case) → 1.0
    - zone's TF matches `primary_sub_tf` (M15 native) → 0.5
    - anything else (assumed main-TF, e.g. H1 overlay) → 0.2

    Returns the multiplier from `STYLE["opacity_tiers.subordinate_chart"]`
    so the values stay centralized in the style registry.
    """
    tiers = STYLE.get("opacity_tiers.subordinate_chart", {})
    if smallest_sub_tf is not None and zone_tf == smallest_sub_tf:
        return float(tiers.get("sub_tf_smallest", 1.0))
    if zone_tf == primary_sub_tf:
        return float(tiers.get("sub_tf", 0.5))
    return float(tiers.get("main_tf", 0.2))
