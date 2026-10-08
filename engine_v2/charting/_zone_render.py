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
from engine_v2.zones.poi_lifecycle import active_stretches_from_history


# ---------------------------------------------------------------------------
# Collapsed-cycle zones (never tradeable) — not drawn (chart review 2026-09-20)
# ---------------------------------------------------------------------------

def is_collapsed_cycle_zone(zone: Any) -> bool:
    """True for a KL zone whose cycle COLLAPSED under its structure's
    lifecycle floor: `status == "inactive"` with a known `end_idx` and the
    clamped `confirmed_idx >= end_idx` (`compute_cycle_lifecycle`: cycle start
    >= cycle end → empty activation_history → "inactive"). Such a zone existed
    geometrically but was never active in real time — on H1 the retroactive
    cycles of a post-reversal sid (sid 1's (1,0)/(1,1) on the reference
    window, the "degenerate parent cycles"); on a sub its forming-phase cycles
    (ended at/before the sub's `start_idx`). Both charts skip them; the rows
    stay in the CSVs. A zone that is inactive for any OTHER reason (a POI that
    never met its activation conditions inside a live window: `end_idx` None
    or later than its start) is NOT collapsed and is still drawn as an outline.
    """
    meta = getattr(zone, "meta", None) or {}
    if meta.get("status") != "inactive":
        return False
    end_idx = meta.get("end_idx")
    conf = meta.get("confirmed_idx")
    if end_idx is None or conf is None:
        return False
    try:
        return int(conf) >= int(end_idx)
    except (TypeError, ValueError):
        return False


def collapsed_cycles(kl_zones: Sequence[Any]) -> set:
    """`{(structure_id, cycle_id)}` of the cycles whose BOS KL zone is
    collapsed (`is_collapsed_cycle_zone`) — the key set used to skip the
    POIs of those cycles (a POI's own `status` cannot distinguish "collapsed
    cycle" from "never activated inside a live cycle")."""
    out = set()
    for z in kl_zones:
        if getattr(z, "source_kind", None) != "BOS":
            continue
        if not is_collapsed_cycle_zone(z):
            continue
        m = z.meta or {}
        sid, cyc = m.get("structure_id"), m.get("cycle_id")
        if sid is not None and cyc is not None:
            out.add((int(sid), int(cyc)))
    return out


def is_poi_of_collapsed_cycle(poi: Any, collapsed: set) -> bool:
    m = getattr(poi, "meta", None) or {}
    sid, cyc = m.get("structure_id"), m.get("cycle_id")
    if sid is None or cyc is None:
        return False
    return (int(sid), int(cyc)) in collapsed


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

    Delegates the pairing walk to `zones.poi_lifecycle` so the chart fills,
    the proximity gate, and the hover labels can never disagree.
    """
    history = zone.meta.get("activation_history", []) or []
    return active_stretches_from_history(history, open_end_idx=render_end_idx)


# ---------------------------------------------------------------------------
# POI hover: the institutional pattern (PLAN_H HV)
# ---------------------------------------------------------------------------

def poi_ip_hover_text(meta: dict, ic_idx: int) -> str:
    """The POI hover's pattern line(s) — ONE builder for the three POI hover
    sites (the H1 chart, the M15 lens charts' sub POIs and their H1 overlay;
    PLAN_H HV): `IP <rank> · legs a-b-c (initial | opp | continuation)` with
    each leg as `first–last` (one candle: its index), + `<br>TB1 over <IC>
    (IC candle type)` / `<br>TB2 over <IC> (reference imbalance <c2>)` when a
    tiebreaker decided. Indices are the POI's own (entity-absolute on a
    mirrored sub POI); the counts must match the leg indices (asserted — a
    missed mirror shift would break it)."""
    n_init, n_opp, n_cont = (int(x) for x in meta["ip_leg_candle_counts"].split("-"))
    anchor, opp_first, end = (int(meta["pattern_anchor_idx"]), int(meta["opp_leg_first_idx"]),
                              int(meta["pattern_end_idx"]))
    ic = int(ic_idx)
    assert (opp_first - anchor, ic - opp_first + 1, end - ic) == (n_init, n_opp, n_cont), (
        f"[chart] POI {ic} legs {meta['ip_leg_candle_counts']} vs indices {anchor}/{opp_first}/{end}")

    def _rng(lo: int, hi: int) -> str:
        return f"{lo}" if lo == hi else f"{lo}–{hi}"

    text = (f"IP {meta['ip_rank']} · legs {n_init}-{n_opp}-{n_cont} "
            f"({_rng(anchor, opp_first - 1)} | {_rng(opp_first, ic)} | {_rng(ic + 1, end)})")
    tb = meta["ip_tiebreak"]
    if tb == "TB1":
        text += f"<br>TB1 over {meta['ip_tb_runner_up_ic_idx']} (IC candle type)"
    elif tb == "TB2":
        text += (f"<br>TB2 over {meta['ip_tb_runner_up_ic_idx']} "
                 f"(reference imbalance {meta['ip_tb_imbalance_first_c2_idx']})")
    else:
        assert tb is None, f"[chart] POI {ic} unknown ip_tiebreak {tb!r}"
    return text


# ---------------------------------------------------------------------------
# Fib records: the draw gate + the hover status cell (both charts)
# ---------------------------------------------------------------------------

def fib_record_drawn(fib_st: Any, collapsed: set) -> bool:
    """The fib draw gate — ONE rule for the H1 chart and the M15 lens charts
    (PLAN_H §8.9 F1, the user 2026-10-06; refines FIB_LIFECYCLE_SPEC §9.1).
    Draw a record iff it is not `disappeared`, its cycle is not COLLAPSED
    (`collapsed` = `collapsed_cycles(...)` of the record's structure: the cycle
    ended before its structure was live — never tradeable, its KL / POI are
    hidden too), and it is locked OR condition-active, ended or not (an ended
    never-locked record was live until its end, e.g. a sub's last cycle ended
    by `parent_end`). `active` False — a superseded cross version, a fib
    invalidated by an imbalance fill — still vanishes (§9.3)."""
    if getattr(fib_st, "status", "active") == "disappeared":
        return False
    if (int(fib_st.structure_id), int(fib_st.cycle_id)) in collapsed:
        return False
    return bool(fib_st.locked or fib_st.active)


def fib_is_bright(fib_st: Any) -> bool:
    """The live version — condition-active, not locked, cycle not ended —
    takes the `fib.*.active` style keys; every other drawn record renders
    faded (`fib.*.historical`)."""
    return bool(fib_st.active and not fib_st.locked
                and getattr(fib_st, "status", "active") == "active")


def ip_rows_by_fib_record(ip_candidates: Sequence[Any], *, by_sub: bool) -> dict:
    """The `attrs["ip_candidates"]` row holding each fib record's IP outcome —
    its picked row or its `IP: none` row (`ic_idx` None) — keyed by
    `(structure_id, cycle_id, bos_anchor_idx, cts_anchor_idx)`, prefixed with
    `meta["sub_id"]` when `by_sub` (an M15 lens: a sub's structure_id restarts
    at 0 in every sub). A record the POI record gate skipped has no row."""
    out: dict = {}
    for r in ip_candidates:
        if not (r.picked or r.ic_idx is None):
            continue
        key = (int(r.structure_id), int(r.cycle_id), int(r.bos_anchor_idx), int(r.cts_anchor_idx))
        if by_sub:
            key = (int(r.meta["sub_id"]),) + key
        assert key not in out, f"[chart] two IP outcome rows for fib record {key}"
        out[key] = r
    return out


def fib_hover_limit_text(fib_st: Any) -> Optional[str]:
    """The fib hover's cross-fib LIMIT line (PLAN_I R6-7) from the record's
    `cross_fib_limit_*` meta — None when the record carries none (a fixture).
    `limit <P> · <source> [(sub <from>)] · M=<M | none>`, or, without `P`,
    `limit: none (<why>)`: the structure's start (`start_trading`, a
    `first_confluence on parent cycle 0`), `no previous sub`, or
    `prev_sub_no_cts (sub <from>)`. Example (sub 7): `limit 0.57827 ·
    prev_sub_bos_threshold (sub 6) · M=2`."""
    m = getattr(fib_st, "meta", None) or {}
    if "cross_fib_limit_source" not in m:
        return None
    source = m["cross_fib_limit_source"]
    price = m["cross_fib_limit_price"]
    frm = m["cross_fib_limit_from_sub_id"]
    sub = f" (sub {int(frm)})" if frm is not None else ""
    if price is None:
        if source == "prev_sub_no_cts":
            why = f"prev_sub_no_cts{sub}"
        elif m.get("started_by") == "start_trading":
            why = "start_trading"
        elif m.get("started_by") == "first_confluence":
            why = f"first_confluence on parent cycle {m.get('parent_cycle_id')}"
        else:
            why = "no previous sub"
        return f"limit: none ({why})"
    M = m["cross_fib_limit_cycle"]
    return f"limit {float(price):.5f} · {source}{sub} · M={'none' if M is None else int(M)}"


def fib_hover_status_text(fib_st: Any, ip_row: Optional[Any]) -> str:
    """The fib hover's status cell — ONE builder for both charts (PLAN_H §8.9
    F1): the lifecycle status (+ ` (locked)`), `life=[start_idx, end_idx]
    end_reason` (`–` = never started, `open` = not ended), the cycle's IP
    outcome (`ip_rows_by_fib_record`): `IP <rank> · IC <idx>` (the pick = the
    drawn POI), `IP: none (<reason>)` (no pattern → no POI, PLAN_H C1), or
    `IP: not evaluated (no POI record)` (no row: the POI record gate skipped
    the record — ended before locking, or no CTS_ESTABLISHED), and the
    structure's cross-fib limit (`fib_hover_limit_text`, PLAN_I R6-7)."""
    start = "–" if fib_st.start_idx is None else int(fib_st.start_idx)
    end = "open" if fib_st.end_idx is None else int(fib_st.end_idx)
    life = f"life=[{start}, {end}]" + (f" {fib_st.end_reason}" if fib_st.end_reason else "")
    if ip_row is None:
        ip = "IP: not evaluated (no POI record)"
    elif ip_row.ic_idx is None:
        ip = f"IP: none ({ip_row.reason})"
    else:
        ip = f"IP {ip_row.ip_rank} · IC {int(ip_row.ic_idx)}"
    status = getattr(fib_st, "status", "active")
    text = f"{status}{' (locked)' if fib_st.locked else ''}<br>{life}<br>{ip}"
    limit = fib_hover_limit_text(fib_st)
    return text if limit is None else f"{text}<br>{limit}"


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
