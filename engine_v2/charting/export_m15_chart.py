"""M15 chart with H1 overlay — separate chart file alongside the H1 chart.

Renders the full M15 dataset as the base candle layer, with all M15 structures
overlaid. Data sourced from `m15_df.attrs["events"] / ["kl_zones"] /
["poi_zones"] / ["fib_states"] / ["wave_candles"] / ["wvmi"]` grouped by each
snapshot's **`sub_id`** (PART4 §17.9 — the unique sub is the chart identity),
with the `m15_df.attrs["sids"]` SidRecord list (one row per unique sub on this
lens) as the manifest and `attrs["triggers"]` (this lens's TriggerRecords) for
hover attribution. Per spec §16.5 (rev 2):

  - Sid-tied elements (CTS/BOS dots, swing lines, PB markers, prev_bos lines;
    the M15 chart renders no WVMI hover): only render at candles the sub OWNS,
    keyed `(candle, direction)` because a +1 and a −1 sub may both be live
    (§17.5). Two layers (chart review 2026-09-20, option 2): the LIVE layer
    `_compute_owner_by_idx_dir` = the sub's real-time lifecycle window
    `[start_idx, end_idx or edge]`, later start wins among live subs; the
    FORMING layer `_compute_forming_by_idx_dir` = the sub's pre-live span
    `[starting_idx, start_idx)`, later anchor wins among forming subs. A sub's
    live elements draw where it is the live owner; its forming elements draw
    where it is the forming owner — even under a live sub of the same direction
    (a structure forming under a live one stays visible; the styles make the
    overlap legible). So a sub's BOS→CTS structure is drawn continuously from
    its structural anchor. Style is decided per SEGMENT by the RECENT-vs-PRIOR
    rule (chart review 2026-09-22, `_prior_line_segments`): where two different
    structures draw lines over the same candles, the MOST RECENT one (higher
    `_recency_key` = `(parent_sid, parent_cycle_id, sub_id)`) is solid and the
    prior one's whole segment is dotted (`structure.m15.*_prior`); a segment no
    other structure overlaps is always solid, whether or not it was ever live in
    real time. Overlap needs more than one shared candle (a shared join candle
    does not count) and ignores direction. Dots follow their segments (filled
    iff they end at least one solid segment). The hover keeps the real-time fact
    `phase=live|forming` (`idx >= start_idx`) and adds `layer=recent|prior`, so
    a dotted line that WAS live in real time is still readable as such — the
    zones carry the real-time lifecycle. (This replaces the 2026-09-21 wave rule
    — solid iff the wave's span touched the lifecycle window — which is still
    what the H1 overlay filter uses.) Prev-BOS lines carry no lifecycle or
    recency formatting at all (always solid, and they never dot anything). A sub
    ended by `same_dir_replacement` runs its final segment through one more
    structural point (`_replacement_break_point`) instead of straight to the
    handover candle. The H1
    OVERLAY's structure lines are lifecycle-FILTERED per wave instead
    (`_render_h1_overlay`: waves never live are not drawn — the H1 chart itself
    is unchanged). Forming-phase ZONES (a KL/POI whose cycle ended at/before
    the sub's `start_idx`, collapsed to `status="inactive"`) are NOT drawn
    (`_zone_render.is_collapsed_cycle_zone`, shared with the H1 chart, which skips
    the collapsed retroactive cycles of a post-reversal sid the same way); they
    stay in the CSVs.
  - Persisting events (KL zones, POI zones, fibs): render all snapshots —
    drawn from the anchor, active from `start_idx` (the KL/POI clamp);
    opacity is a per-TF tier (`_m15_opacity_tier_for_zone`).
  - Every lens draws a sub over the SAME window (the sub's), so a sub on both
    charts looks identical on each.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go

from engine_v2.common.types import COL_C, COL_H, COL_L, COL_O, COL_TIME, COL_V, PatternStatus
from engine_v2.charting.style_registry import STYLE
from engine_v2.structure import event_fields as ef
from engine_v2.charting._zone_render import (
    build_stepped_outline_xy,
    collapsed_cycles,
    compute_kl_active_stretches,
    compute_poi_active_stretches,
    is_collapsed_cycle_zone,
    is_poi_of_collapsed_cycle,
    select_subordinate_tf_tier,
)
from engine_v2.zones.poi_lifecycle import poi_confirmed_idx_as_of
from engine_v2.zones.structure_lifecycle import (
    compute_reversal_idx_by_sid,
    compute_struct_start_by_sid,
)
from engine_v2.charting.export_plotly import (
    _rgba_from_rgb,
    _zone_style,
    _style,
    _opacity_tier,
    _deep_merge,
    _get_reversal_confirmed_by_sid,
    ChartExportPaths,
)
from engine_v2.multitf.registry import StructureRegistry
from engine_v2.multitf.types import SidRecord


# ---------------------------------------------------------------------------
# M15 chart config defaults
# ---------------------------------------------------------------------------
M15_CHART_DEFAULTS = {
    "show_ohlc": True,
    "candle_types": {},
    "patterns": {
        "continuous": True,
        "double_maru": True,
        "one_maru_continuous": True,
        "one_maru_opposite": True,
    },
    "struct_state": {"labels": False},
    "range_visual": {"rectangles": False},
    "structure": {"levels": True},
    "zones": {"KL": True, "POI": True, "wave_candles": True, "num_structures": 99},
    "fib": {"lines": False},
    "imbalance": {"highlight": True},
    "volume": {"bars": True, "ema_line": True, "spike_marker": True},
    "range_candle_marker": False,
}


# ---------------------------------------------------------------------------
# H1→M15 time mapping helpers
# ---------------------------------------------------------------------------

def _build_h1_to_m15_map(
    h1_times: pd.Series,
    m15_times: pd.Series,
) -> dict:
    """Map each H1 candle time to the 4th (last) M15 candle in that H1 hour.

    Returns {h1_time → m15_time}.
    Fallback: h1_time + 45 min if no M15 candle in that hour.
    """
    m15_by_hour: dict = {}
    for t in m15_times:
        hour_key = t.floor("h")
        m15_by_hour.setdefault(hour_key, []).append(t)

    mapping = {}
    for h1t in h1_times:
        h1_hour = h1t.floor("h") if hasattr(h1t, "floor") else pd.Timestamp(h1t, tz="UTC").floor("h")
        candidates = m15_by_hour.get(h1_hour, [])
        if candidates:
            mapping[h1t] = max(candidates)  # Last M15 in that hour
        else:
            mapping[h1t] = h1t + timedelta(minutes=45)
    return mapping


def _build_m15_to_h1_map(
    h1_df: pd.DataFrame,
    m15_times: pd.Series,
) -> dict:
    """Map each M15 candle time to its parent H1 candle's (time, idx).

    Returns {m15_time → (h1_time, h1_idx)}.
    Uses m15_time.floor('h') to find the parent H1 hour.
    """
    h1_times = pd.to_datetime(h1_df[COL_TIME], utc=True)
    h1_by_hour: dict = {}
    for idx, t in zip(h1_df.index, h1_times):
        hour_key = t.floor("h")
        h1_by_hour[hour_key] = (t, int(idx))

    mapping = {}
    for m15t in m15_times:
        m15_hour = m15t.floor("h") if hasattr(m15t, "floor") else pd.Timestamp(m15t, tz="UTC").floor("h")
        if m15_hour in h1_by_hour:
            mapping[m15t] = h1_by_hour[m15_hour]
        else:
            mapping[m15t] = (None, None)
    return mapping


# ---------------------------------------------------------------------------
# M15 opacity tier helpers
# ---------------------------------------------------------------------------

def _m15_opacity_tier_for_zone(
    zone,
    primary_sub_tf: str = "M15",
) -> float:
    """Per-TF tier multiplier for an M15-chart zone (Item 5, 2026-05-20).

    Replaces the prior active/recent_inactive/prior_inactive 3-tier system
    with a per-TF tier (Spec 2): zones from the chart's primary sub-TF (M15)
    get the `sub_tf` multiplier (0.5), main-TF overlays (H1) get `main_tf`
    (0.2). A zone's end_time is capped at its sub's lifecycle end by the
    projection (`render_sub_projection`: reversal | same_dir_replacement |
    parent_end), so its visible extent is already truncated to that boundary
    regardless of tier.

    Zone's TF comes from `meta["timeframe"]`; defaults to `primary_sub_tf`
    when missing (the M15-native rendering path always emits M15 zones).
    """
    zone_tf = zone.meta.get("timeframe", primary_sub_tf)
    return select_subordinate_tf_tier(zone_tf, primary_sub_tf=primary_sub_tf)


def _m15_opacity_tier_for_events(
    parent_sid: int,
    parent_cycle_id: int,
    most_recent_parent_sid: int,
    recent_cycle_ids: set,
    is_active_sid: bool,
) -> float:
    """Compute 3-tier opacity for M15 structure elements."""
    if is_active_sid:
        return _opacity_tier("active")
    if parent_sid == most_recent_parent_sid and parent_cycle_id in recent_cycle_ids:
        return _opacity_tier("recent_inactive")
    return _opacity_tier("prior_inactive")


def _compute_m15_tier_context_from_sids(
    sid_records: Iterable[SidRecord],
) -> tuple:
    """Compute most_recent_parent_sid and recent_cycle_ids across SidRecords.

    Sourced from each sub's FIRST record (`_sid_parent`, §17.9 — parent fields
    on a sub SidRecord are None); semantics unchanged from the per-trigger era.
    """
    parents = [_sid_parent(s) for s in sid_records]
    parent_sids = [ps for (ps, _pc) in parents if ps is not None]
    if not parent_sids:
        return (0, set())
    most_recent = max(parent_sids)
    cycles_for_recent = sorted(
        [pc for (ps, pc) in parents if ps == most_recent and pc is not None],
        reverse=True,
    )
    recent_cycle_ids = set(cycles_for_recent[:2])
    return (most_recent, recent_cycle_ids)


def _sub_identity(meta: dict) -> Optional[int]:
    """The unique sub's `sub_id` from a snapshot meta dict, or None if the
    snapshot carries no `sub_id` attribution (§17.9 — `sub_id` IS the
    identity; the informational `parent_sid` / `parent_cycle_id` are never
    used for grouping)."""
    sid = meta.get("sub_id")
    return int(sid) if sid is not None else None


def _sid_record_identity(rec: SidRecord) -> int:
    """Chart identity of a SidRecord: `sub_id` for a sub row, `sub_sid`
    (= structure_id) for a main row."""
    if rec.sub_id is not None:
        return int(rec.sub_id)
    return int(rec.sub_sid)


def _sid_parent(rec: SidRecord) -> tuple:
    """`(parent_sid, parent_cycle_id)` for hover/tier context: the record's own
    fields for main rows, else the sub's FIRST record (`meta["first_record"]`,
    informational only)."""
    if rec.parent_sid is not None or rec.parent_cycle_id is not None:
        return (
            int(rec.parent_sid) if rec.parent_sid is not None else None,
            int(rec.parent_cycle_id) if rec.parent_cycle_id is not None else None,
        )
    fr = (rec.meta or {}).get("first_record") or {}
    ps, pc = fr.get("parent_sid"), fr.get("parent_cycle_id")
    return (
        int(ps) if ps is not None else None,
        int(pc) if pc is not None else None,
    )


def _compute_forming_by_idx_dir(sid_records: Iterable[SidRecord]) -> dict:
    """Per-(candle, direction) FORMING owner map: each sub's pre-live span
    `[creation_event_idx (= starting_idx, the structural anchor), start_idx)`,
    later anchor wins among forming subs. Independent of the LIVE map
    (`_compute_owner_by_idx_dir`): a sub's forming elements draw where it is
    the forming owner even under a live sub of the same direction (the
    forming style is visually subordinate). Rows with `start_idx is None` or
    `creation_event_idx is None` are skipped."""
    owner: dict = {}
    rows = [r for r in sid_records if r.start_idx is not None and r.creation_event_idx is not None]
    for rec in sorted(rows, key=lambda r: (int(r.creation_event_idx), _sid_record_identity(r))):
        ident = _sid_record_identity(rec)
        d = int(rec.starting_sd)
        for i in range(int(rec.creation_event_idx), int(rec.start_idx)):
            owner[(i, d)] = ident
    return owner


def _compute_owner_by_idx_dir(sid_records: Iterable[SidRecord], edge_idx: int) -> dict:
    """Per-(candle, direction) owner map for the §16.5 sid-tied display rule
    (rev 2): each sub owns `[start_idx, end_idx or edge_idx]` — its REAL-TIME
    lifecycle window, not its structural anchor — keyed by `(candle,
    direction)` because opposite-direction subs may both be live on one chart
    (§17.5). Walked in `(start_idx, identity)` order so a later start wins a
    same-direction overlap. Rows with `start_idx is None` are skipped (a sub
    with no live record is logged, not rendered)."""
    owner: dict = {}
    rows = [r for r in sid_records if r.start_idx is not None]
    for rec in sorted(rows, key=lambda r: (int(r.start_idx), _sid_record_identity(r))):
        ident = _sid_record_identity(rec)
        end = int(rec.end_event_idx) if rec.end_event_idx is not None else int(edge_idx)
        d = int(rec.starting_sd)
        for i in range(int(rec.start_idx), end + 1):
            owner[(i, d)] = ident
    return owner


def _h1_overlay_window_start_by_sid(events, reversal_idx_by_sid) -> dict:
    """The H1 overlay's per-sid window START on the sub charts — a LOCATION
    (user decision 2026-09-25, Plan E E3f landing review): the overlay draws
    structure geometry anchor-to-anchor, so a sid's window opens at its first
    structural ANCHOR (min `ef.stamped_idx` — BOS_0's) and its defining first
    leg is always shown, even when CTS_0 lags its moment; a sid N >= 1 whose
    predecessor reversed opens at that reversal (the handoff — its retroactive
    legs before it stay hidden, chart review 2026-09-21). Not
    `compute_struct_start_by_sid`, whose base is the first CTS_ESTABLISHED
    MOMENT (a lifecycle TIME)."""
    start = {}
    for ev in events:
        s = (ev.meta or {}).get("structure_id")
        if s is None:
            continue
        s = int(s)
        i = ef.stamped_idx(ev)
        if s not in start or i < start[s]:
            start[s] = i
    for s in list(start):
        if (s - 1) in reversal_idx_by_sid:
            start[s] = int(reversal_idx_by_sid[s - 1])
    return start


def _wave_touches_window(
    a_idx: int, b_idx: int, start_idx: Optional[int], end_idx: Optional[int],
) -> bool:
    """Wave rule (chart review 2026-09-21). A WAVE is one straight segment of a
    sid-tied line between candles `a_idx` and `b_idx` — the EXTREME candles the
    line is drawn through (`cts_anchor_idx` / `bos_anchor_idx`), not the confirmation
    candles. It was live at some point iff its candle span intersects the
    structure's real-time lifecycle window `[start_idx, end_idx]` (`end_idx`
    None = open): `max(a,b) >= start_idx and (end_idx is None or min(a,b) <=
    end_idx)`. A structure with no `start_idx` (never live) has no live wave.
    Used for the sub charts' solid-vs-forming style AND the H1 overlay's
    drawn-vs-hidden filter."""
    if start_idx is None:
        return False
    lo, hi = (int(a_idx), int(b_idx)) if int(a_idx) <= int(b_idx) else (int(b_idx), int(a_idx))
    if hi < int(start_idx):
        return False
    return end_idx is None or lo <= int(end_idx)


def _group_flag_runs(flags: Sequence[bool]) -> list:
    """Group per-SEGMENT flags into point-index runs `[(flag, i0, i1), ...]` —
    the run is drawn through points `i0..i1` (inclusive, `i1 > i0`), `flags[i]`
    being the flag of the segment between points `i` and `i+1`. Adjacent runs
    alternate; no flags → `[]`. The shared point belongs to BOTH neighbouring
    runs, so a style change never leaves a gap."""
    n = len(flags)
    if n == 0:
        return []
    runs: list = []
    i0 = 0
    for i in range(1, n):
        if flags[i] != flags[i0]:
            runs.append((flags[i0], i0, i))
            i0 = i
    runs.append((flags[i0], i0, n))
    return runs


def _split_polyline_by_wave(
    idx_seq: Sequence[int], start_idx: Optional[int], end_idx: Optional[int],
) -> list:
    """Split a polyline (point candle idxs in drawing order) into runs of
    consecutive waves that share one live flag (`_wave_touches_window` per
    wave, `_group_flag_runs` over the flags). An ENTIRE wave counts as live
    when any part of it was. Used by the H1 OVERLAY filter (never-live waves
    are not drawn); the sub charts style their segments by
    `_prior_line_segments` instead."""
    n = len(idx_seq)
    if n < 2:
        return []
    return _group_flag_runs(
        [_wave_touches_window(idx_seq[i], idx_seq[i + 1], start_idx, end_idx) for i in range(n - 1)]
    )


def _recency_key(rec: SidRecord) -> tuple:
    """Display recency of a structure: the hierarchical `(parent_sid,
    parent_cycle_id, sub_id)` tuple (chart review 2026-09-22). Of two
    structures drawing over the same candles the later one always has the
    higher tuple. The first two components come from the sub's FIRST record
    (`_sid_parent`, informational) and the last is the canonical monotonic
    identity, so the tuple orders exactly like `sub_id` whenever the parents
    agree — which they do on every window measured (triggers fire in
    parent-cycle order). Missing parents sort first (`-1`)."""
    ps, pc = _sid_parent(rec)
    return (
        int(ps) if ps is not None else -1,
        int(pc) if pc is not None else -1,
        _sid_record_identity(rec),
    )


def _prior_line_segments(segments_by_sub: dict) -> set:
    """RECENT-vs-PRIOR rule (chart review 2026-09-22). Input
    `{sub_key: (recency_key, [(seg_id, lo_idx, hi_idx), ...])}` over every
    sid-tied line segment DRAWN ON THIS LENS (each wave of each polyline, the
    extension to the last owned candle, and each PB→BOS line; prev-BOS lines
    do not take part). Returns `{(sub_key, seg_id)}` — the segments to draw in
    the PRIOR (dotted) style because a structure with a HIGHER `recency_key`
    draws a segment over the same candles:

        overlap  ⇔  `lo_a < hi_b and lo_b < hi_a`

    i.e. MORE THAN ONE shared candle — two segments meeting at a single join
    candle are not an overlap. Direction-agnostic: a `+1` and a `−1` structure
    crowding the same candles are still ordered (the newer wins). Formatting is
    per WHOLE segment: a segment that overlaps for even part of its span is
    dotted end to end. Everything not returned is drawn solid — including
    structures that were never live in real time, which no other structure
    overlaps. Per lens: a sub can be prior on one chart and solid on the other
    (the other lens may not draw the structure that supersedes it)."""
    out: set = set()
    items = list(segments_by_sub.items())
    for key_a, (ord_a, segs_a) in items:
        for key_b, (ord_b, segs_b) in items:
            if key_b == key_a or not (ord_b > ord_a):
                continue
            for seg_id, lo_a, hi_a in segs_a:
                if (key_a, seg_id) in out:
                    continue
                for _seg_b, lo_b, hi_b in segs_b:
                    if lo_a < hi_b and lo_b < hi_a:
                        out.add((key_a, seg_id))
                        break
    return out


def _make_sub_phase_fns(eid: int, sub_dir: int, sub_start: Optional[int],
                        live_by_idx_dir: dict, forming_by_idx_dir: dict):
    """`(is_live, owned_here)` for one sub. `is_live(idx)` is the real-time
    fact (`idx >= start_idx`, feeds the hover `phase`); `owned_here(idx)` is
    the §16.5 sid-tied filter that decides whether this sub draws at that
    candle at all — the LIVE map inside its lifecycle window, the FORMING map
    before it (see the two ownership layers above). Style is decided
    separately by `_prior_line_segments`."""
    def is_live(idx: int) -> bool:
        return sub_start is not None and int(idx) >= sub_start

    def owned_here(idx: int) -> bool:
        k = (int(idx), sub_dir)
        if is_live(idx):
            return live_by_idx_dir.get(k) == eid
        return forming_by_idx_dir.get(k) == eid

    return is_live, owned_here


def _replacement_break_point(
    sid_rec, sub_records, anchors_by_sub, lt_df, last_pt_idx: int, ext_end_idx: int,
):
    """The extra structural point a REPLACED sub's final segment runs through
    (chart review 2026-09-22b), or None.

    A sub ended by `same_dir_replacement` is superseded while price keeps
    moving, so its line is dragged from its last confirmed point to the handover
    candle — straight through a real swing extreme. Every other end reason
    already ends at one (measured on the reference window: the four
    reversal-ended subs' extensions end 0–2 candles from their counter-move
    extreme; `parent_end` ends at the parent's candle), so the break applies to
    replacement only.

    The point is the extreme of the COUNTER-move (a `−1` sub pulls up → highest
    high; a `+1` sub pulls down → lowest low — the same low/high convention as
    the existing pullback dots) over `(last_pt_idx, hi]`, where `hi` is the
    REPLACING structure's anchor (`starting_idx`, clamped to the drawn end).
    Bounding at the anchor is what makes it the STRUCTURAL swing rather than a
    later marginal overshoot: on the reference window sub `3304/−1` breaks at
    3760 (0.57806, where the sibling structures put the swing) and not at the
    literal highest high 3806 (0.57827, 2.1 pips higher — MS saw it, see sub
    `3621/+1`'s `CTS_THRESHOLD_UPDATED@3806`, and kept the swing at 3760). The
    resulting segments mirror the sibling structures point for point: sub
    `3304/−1`'s 3621→3760 is sub `3621/+1`'s segment, sub `3621/+1`'s 3760→4000
    is sub `3760/−1`'s.

    The replacing sub comes from the record that ended this sub
    (`TriggerRecord.ended_by_sub_id`; same lens by construction — a counter
    record never ends a confluence one) and its anchor from `anchors_by_sub`.
    Returns `(idx, price)` strictly inside `(last_pt_idx, ext_end_idx)`, else
    None."""
    if (getattr(sid_rec, "end_reason", None) or "") != "same_dir_replacement":
        return None
    sub_end = sid_rec.end_event_idx
    repl_id = None
    for tr in sub_records or ():
        if getattr(tr, "end_reason", None) != "same_dir_replacement":
            continue
        if getattr(tr, "ended_by_sub_id", None) is None:
            continue
        if sub_end is not None and tr.end_idx is not None and int(tr.end_idx) != int(sub_end):
            continue
        repl_id = int(tr.ended_by_sub_id)
        break
    if repl_id is None:
        return None
    anchor = anchors_by_sub.get(repl_id)
    if anchor is None:
        return None

    lo = int(last_pt_idx) + 1
    hi = min(int(anchor), int(ext_end_idx))
    if hi < lo:
        return None
    col = COL_H if int(sid_rec.starting_sd) == -1 else COL_L
    best_idx, best_price = None, None
    for i in range(lo, hi + 1):
        if i >= len(lt_df):
            break
        price = float(lt_df.iloc[i][col])
        if best_price is None or (price > best_price if col == COL_H else price < best_price):
            best_idx, best_price = i, price
    if best_idx is None or not (int(last_pt_idx) < best_idx < int(ext_end_idx)):
        return None
    return best_idx, best_price


def _build_sub_polylines(sid_rec, sid_events, lt_df, lt_time, lt_full_idx, owned_here,
                         sub_records=None, anchors_by_sub=None) -> dict:
    """Build one sub's sid-tied POLYLINES (no drawing) so every sub's segments
    exist before any is styled (`_prior_line_segments` compares across subs).

    Returns `points_by_sid` (per internal MS sid, the confirmed CTS/BOS points
    plus the trailing unconfirmed-CTS or pullback point), the `extra_cts_pts` /
    `extra_pb_pts` dot lists, the cross-structure `pb_to_bos_lines`
    (`… , pb_idx, bos_idx`), `extend_to_idx` (the sub's lifecycle end walked
    back to the last candle it still owns), `most_recent_lt_sid`, and
    `seq_by_sid` — the drawn point sequence per internal sid, with the
    extension appended as an `"EXT"` pseudo-point on the most recent one.
    Every point is filtered by `owned_here` at the candle it is DRAWN at
    (`cts_anchor_idx` for CTS, `bos_anchor_idx` for BOS)."""
    cts_events = [e for e in sid_events if e.type == "CTS_CONFIRMED"]
    bos_events = [e for e in sid_events if e.type == "BOS_CONFIRMED"]
    all_sids_lt = set()
    for e in cts_events + bos_events:
        all_sids_lt.add(int(e.meta.get("structure_id", 0)))
    most_recent_lt_sid = max(all_sids_lt) if all_sids_lt else 0

    points_by_sid = defaultdict(list)
    for ev in cts_events:
        p_idx = ef.cts_anchor_idx(ev)
        if not owned_here(p_idx):
            continue
        t = lt_time(p_idx)
        if t is None:
            continue
        price = float(ev.price) if ev.price is not None else 0.0
        if price == 0.0 and "cts_price" in lt_df.columns and p_idx < len(lt_df):
            price = float(lt_df.iloc[p_idx]["cts_price"]) if not pd.isna(lt_df.iloc[p_idx].get("cts_price", float("nan"))) else 0.0
        sid = int(ev.meta.get("structure_id", 0))
        cycle = int(ev.meta.get("cycle_id", 0))
        sd = int(ev.meta.get("struct_direction", 0))
        full_idx = lt_full_idx(p_idx)
        points_by_sid[sid].append((p_idx, t, price, "CTS", sid, cycle, sd, full_idx))

    for ev in bos_events:
        b_idx = ef.bos_anchor_idx(ev)   # the BOS dot sits at its anchor
        if not owned_here(b_idx):
            continue
        t = lt_time(b_idx)
        if t is None:
            continue
        price = float(ev.price) if ev.price is not None else 0.0
        sid = int(ev.meta.get("structure_id", 0))
        cycle = int(ev.meta.get("cycle_id", 0))
        sd = int(ev.meta.get("struct_direction", 0))
        full_idx = lt_full_idx(b_idx)
        points_by_sid[sid].append((b_idx, t, price, "BOS", sid, cycle, sd, full_idx))

    # Unconfirmed CTS + PB dots
    cts_unconf = [e for e in sid_events if e.type in ("CTS_ESTABLISHED", "CTS_UPDATED")]
    pb_events = [e for e in sid_events if e.type == "STATE_CHANGED" and e.meta.get("to") == "pullback"]

    extra_cts_pts = []
    extra_pb_pts = []
    pb_to_bos_lines = []

    for sid in sorted(all_sids_lt):
        sid_pts = sorted(points_by_sid.get(sid, []), key=lambda x: x[0])
        if not sid_pts:
            continue
        last_pt = sid_pts[-1]
        last_kind = last_pt[3]
        last_slice_idx = last_pt[0]
        sd_for_sid = last_pt[6]

        if last_kind == "BOS":
            # The dot sits at the CTS ANCHOR (x) with ev.price (y) — a location.
            cts_after = [e for e in cts_unconf
                         if int(e.meta.get("structure_id", -1)) == sid
                         and ef.cts_anchor_idx(e) > last_slice_idx
                         and owned_here(ef.cts_anchor_idx(e))]
            if cts_after:
                latest = max(cts_after, key=ef.cts_anchor_idx)
                latest_idx = ef.cts_anchor_idx(latest)
                t = lt_time(latest_idx)
                if t is not None:
                    price = float(latest.price) if latest.price is not None else 0.0
                    cycle = int(latest.meta.get("cycle_id", 0))
                    full_idx = lt_full_idx(latest_idx)
                    kind_label = latest.type.replace("CTS_", "").lower()
                    points_by_sid[sid].append((latest_idx, t, price, "CTS", sid, cycle, sd_for_sid, full_idx))
                    extra_cts_pts.append((latest_idx, t, price, f"CTS ({kind_label})", sid, cycle, sd_for_sid, full_idx))

        elif last_kind == "CTS" and sid != most_recent_lt_sid:
            next_sid = sid + 1
            next_bos_evs = sorted(
                [e for e in bos_events if int(e.meta.get("structure_id", -1)) == next_sid],
                key=ef.event_moment,
            )
            # The PB search's upper bound is a TIME: the next sid's first BOS MOMENT
            # (Plan E E3g-3, PLAN_E §7.1 T4).
            next_bos_idx = ef.event_moment(next_bos_evs[0]) if next_bos_evs else None

            pb_after = [e for e in pb_events
                        if int(e.meta.get("structure_id", -1)) == sid
                        and int(e.idx) > last_slice_idx   # a LOCATION lower bound: PBs after the last point's extreme
                        and (next_bos_idx is None or int(e.idx) < next_bos_idx)
                        and owned_here(e.idx)]
            if pb_after:
                latest_pb = max(pb_after, key=lambda e: int(e.idx))
                t = lt_time(latest_pb.idx)
                if t is not None:
                    if sd_for_sid == 1:
                        pb_price = float(lt_df.iloc[latest_pb.idx][COL_L])
                    else:
                        pb_price = float(lt_df.iloc[latest_pb.idx][COL_H])
                    full_idx = lt_full_idx(latest_pb.idx)
                    points_by_sid[sid].append((latest_pb.idx, t, pb_price, "PB", sid, 0, sd_for_sid, full_idx))
                    extra_pb_pts.append((latest_pb.idx, t, pb_price, "PB", sid, 0, sd_for_sid, full_idx))
                    if next_bos_evs:
                        fb = next_bos_evs[0]
                        fb_idx = ef.bos_anchor_idx(fb)   # the line's BOS end (location)
                        fb_t = lt_time(fb_idx)
                        fb_price = float(fb.price) if fb.price is not None else 0.0
                        if fb_t is not None:
                            pb_to_bos_lines.append((sid, t, pb_price, fb_t, fb_price,
                                                    int(latest_pb.idx), fb_idx))

    # The most-recent internal sid's line extends to this sub's lifecycle end,
    # walked back to the last candle the sub still owns (where a later
    # same-direction sub takes over).
    extend_to_idx = sid_rec.end_event_idx if sid_rec.end_event_idx is not None else (len(lt_df) - 1)
    while extend_to_idx > 0 and not owned_here(extend_to_idx):
        extend_to_idx -= 1

    # A sub ended by `same_dir_replacement` runs its final segment through the
    # last structural swing before the replacement (`_replacement_break_point`)
    # instead of straight to the handover candle — one more wave, drawn with a
    # PB dot (the same low/high convention as the pullback dots above).
    _last_seq = sorted(points_by_sid.get(most_recent_lt_sid, []), key=lambda x: x[0])
    if _last_seq and anchors_by_sub is not None:
        _bp = _replacement_break_point(
            sid_rec, sub_records, anchors_by_sub, lt_df, int(_last_seq[-1][0]), int(extend_to_idx),
        )
        if _bp is not None:
            _b_idx, _b_price = _bp
            _b_t = lt_time(_b_idx)
            if _b_t is not None:
                _pt = (_b_idx, _b_t, _b_price, "PB", most_recent_lt_sid, 0,
                       int(sid_rec.starting_sd), lt_full_idx(_b_idx))
                points_by_sid[most_recent_lt_sid].append(_pt)
                extra_pb_pts.append(_pt)

    seq_by_sid: dict = {}
    for sid, pts in points_by_sid.items():
        seq = sorted(pts, key=lambda x: x[0])
        if sid == most_recent_lt_sid and 0 <= extend_to_idx < len(lt_df) and seq and extend_to_idx > seq[-1][0]:
            end_time = lt_time(extend_to_idx)
            if end_time is not None:
                end_price = float(lt_df.iloc[extend_to_idx][COL_C])
                seq.append((extend_to_idx, end_time, end_price, "EXT", sid, -1,
                            int(sid_rec.starting_sd), extend_to_idx))
        seq_by_sid[sid] = seq

    return {
        "points_by_sid": points_by_sid,
        "extra_cts_pts": extra_cts_pts,
        "extra_pb_pts": extra_pb_pts,
        "pb_to_bos_lines": pb_to_bos_lines,
        "extend_to_idx": extend_to_idx,
        "most_recent_lt_sid": most_recent_lt_sid,
        "seq_by_sid": seq_by_sid,
    }


def _segments_of(poly: dict) -> list:
    """`[(seg_id, lo_idx, hi_idx)]` for one sub's drawn line segments — one per
    wave of each internal sid's polyline (`("w", sid, i)`) plus each PB→BOS
    line (`("pb", k)`). The input for `_prior_line_segments`."""
    segs = []
    for sid, seq in poly["seq_by_sid"].items():
        for i in range(len(seq) - 1):
            a, b = int(seq[i][0]), int(seq[i + 1][0])
            segs.append((("w", sid, i), min(a, b), max(a, b)))
    for k, ln in enumerate(poly["pb_to_bos_lines"]):
        a, b = int(ln[5]), int(ln[6])
        segs.append((("pb", k), min(a, b), max(a, b)))
    return segs


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_m15_chart_plotly(
    *,
    title: str,
    registry: StructureRegistry,
    path_id: str,
    out_dir: str | Path = "artifacts/charts",
    basename: str = "chart_m15",
    max_points: Optional[int] = None,
    idx_range: Optional[tuple[int, int]] = None,
    cfg: Optional[dict] = None,
) -> ChartExportPaths:
    """Export an interactive M15 chart with H1 overlay elements.

    Data is read directly from `m15_df.attrs["events" / "kl_zones" /
    "poi_zones" / "fib_states" / "wave_candles" / "wvmi"]` and grouped by each
    snapshot's `sub_id` per the `m15_df.attrs["sids"]` SidRecord manifest
    (one row per unique sub on this lens, §17.9).

    The chart resolves its M15 entity (and its parent for the overlay) via
    ``registry`` + ``path_id`` (§13.5.e: the legacy positional
    ``m15_df`` / ``h1_df`` fallback was removed — this is now the only entry).
    Per spec §16.3 each chart overlays only its immediate parent.
    """

    m15_entity = registry.get(path_id)
    if m15_entity is None:
        raise ValueError(f"StructureRegistry has no entity '{path_id}'")
    parent_entity = registry.parent_of(path_id)
    if parent_entity is None:
        raise ValueError(
            f"M15 chart entity '{path_id}' has no parent in registry")
    m15_df = m15_entity.df
    h1_df = parent_entity.df

    cfg = _deep_merge(M15_CHART_DEFAULTS, cfg or {})
    pat_cfg = cfg.get("patterns", {}) or {}
    struct_cfg = cfg.get("structure", {}) or {}
    zone_cfg = cfg.get("zones", {}) or {}
    imbalance_cfg = cfg.get("imbalance", {}) or {}
    volume_cfg = cfg.get("volume", {}) or {}
    state_cfg = cfg.get("struct_state", {}) or {}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfx = m15_df.copy()

    # Optional index-range slicing
    if idx_range is not None:
        i0, i1 = idx_range
        if i0 > i1:
            i0, i1 = i1, i0
        i0 = max(int(i0), int(dfx.index.min()))
        i1 = min(int(i1), int(dfx.index.max()))
        dfx = dfx.loc[i0:i1].copy()

    if max_points is not None and len(dfx) > max_points:
        dfx = dfx.iloc[-max_points:].copy()

    dfx[COL_TIME] = pd.to_datetime(dfx[COL_TIME], utc=True)

    # Build time mapping helpers
    m15_times = dfx[COL_TIME]
    h1_times = pd.to_datetime(h1_df[COL_TIME], utc=True)
    h1_to_m15 = _build_h1_to_m15_map(h1_times, m15_times)
    m15_to_h1 = _build_m15_to_h1_map(h1_df, m15_times)

    # M15 entity-wide attrs reads. Each list is the union across all subs on
    # this lens; we group by `sub_id` per SidRecord below for per-sub rendering.
    sid_records: list = list(m15_df.attrs.get("sids", []))
    trigger_records: list = list(m15_df.attrs.get("triggers", []))
    records_by_sub: dict = defaultdict(list)
    for tr in trigger_records:
        records_by_sub[int(tr.sub_id)].append(tr)
    all_events: list = list(m15_df.attrs.get("events", []))
    all_kl_zones: list = list(m15_df.attrs.get("kl_zones", []))
    all_poi_zones: list = list(m15_df.attrs.get("poi_zones", []))
    all_fib_states: list = list(m15_df.attrs.get("fib_states", []))
    all_wave_candles: list = list(m15_df.attrs.get("wave_candles", []))
    all_wvmi_records: list = list(m15_df.attrs.get("wvmi", []))
    all_prev_bos_lines: list = list(m15_df.attrs.get("prev_bos_lines", []))

    # Group by sub_id. Snapshots without a sub_id attribution are ignored —
    # they belong to no sub-build (defensive; the mirror always stamps it,
    # but partial entity dfs in tests may not).
    events_by_sid: dict = defaultdict(list)
    kls_by_sid: dict = defaultdict(list)
    pois_by_sid: dict = defaultdict(list)
    fibs_by_sid: dict = defaultdict(list)
    waves_by_sid: dict = defaultdict(list)
    wvmis_by_sid: dict = defaultdict(list)
    prev_bos_by_sid: dict = defaultdict(list)
    for ev in all_events:
        ident = _sub_identity(ev.meta)
        if ident is not None:
            events_by_sid[ident].append(ev)
    for z in all_kl_zones:
        ident = _sub_identity(z.meta)
        if ident is not None:
            kls_by_sid[ident].append(z)
    for p in all_poi_zones:
        ident = _sub_identity(p.meta)
        if ident is not None:
            pois_by_sid[ident].append(p)
    for f in all_fib_states:
        ident = _sub_identity(f.meta)
        if ident is not None:
            fibs_by_sid[ident].append(f)
    for w in all_wave_candles:
        ident = _sub_identity(w.meta)
        if ident is not None:
            waves_by_sid[ident].append(w)
    for r in all_wvmi_records:
        ident = _sub_identity(r.meta)
        if ident is not None:
            wvmis_by_sid[ident].append(r)
    for ln in all_prev_bos_lines:
        ident = _sub_identity(ln.get("meta") or {}) if isinstance(ln, dict) else None
        if ident is not None:
            prev_bos_by_sid[ident].append(ln)

    # §16.5 sid-tied filter: owner (sub_id) per (candle, direction) — the LIVE
    # layer (lifecycle window) wins; the FORMING layer (anchor → start_idx)
    # fills where no same-direction sub is live.
    live_by_idx_dir = _compute_owner_by_idx_dir(sid_records, int(m15_df.index[-1]))
    forming_by_idx_dir = _compute_forming_by_idx_dir(sid_records)

    # M15 opacity tier context (parent_sid + recent cycles). Same semantics
    # as before, just sourced from SidRecord meta rather than trigger meta.
    m15_most_recent_psid, m15_recent_cycles = _compute_m15_tier_context_from_sids(sid_records)

    volume_enabled = volume_cfg.get("bars", False) and COL_V in dfx.columns
    fig = go.Figure()

    wick_offset = (dfx[COL_H] - dfx[COL_L]) * 0.15

    # ---------------------------------------------------------------
    # Helper columns
    # ---------------------------------------------------------------
    candle_idx = dfx.index.to_numpy()

    range_break_frac = (
        dfx["range_break_frac"].astype(float)
        if "range_break_frac" in dfx.columns
        else pd.Series([float("nan")] * len(dfx), index=dfx.index)
    )

    def _col_or_default(name, default):
        return dfx[name] if name in dfx.columns else pd.Series([default] * len(dfx), index=dfx.index)

    is_big_normal_as0 = _col_or_default("is_big_normal_as0", False)
    is_big_maru_as0 = _col_or_default("is_big_maru_as0", False)
    big_ratio_as0 = _col_or_default("big_ratio_as0", 0.0).astype(float)
    vol_spike_ratio = _col_or_default("vol_spike_ratio", float("nan")).astype(float)

    # Build idx_1H and time_1H columns for hover
    idx_1h_col = []
    time_1h_col = []
    for t in dfx[COL_TIME]:
        h1_info = m15_to_h1.get(t, (None, None))
        idx_1h_col.append(h1_info[1] if h1_info[1] is not None else "")
        time_1h_col.append(str(h1_info[0]) if h1_info[0] is not None else "")

    customdata = list(zip(
        candle_idx,                       # 0
        dfx["mid_price"].astype(float) if "mid_price" in dfx.columns
        else ((dfx[COL_H] + dfx[COL_L]) / 2).astype(float),  # 1
        dfx["body_pct"].astype(float) if "body_pct" in dfx.columns
        else pd.Series([0.0] * len(dfx), index=dfx.index),    # 2
        dfx["candle_type"].astype(str) if "candle_type" in dfx.columns
        else pd.Series([""] * len(dfx), index=dfx.index),     # 3
        dfx["body_len"].astype(float) if "body_len" in dfx.columns
        else pd.Series([0.0] * len(dfx), index=dfx.index),    # 4
        dfx["candle_len"].astype(float) if "candle_len" in dfx.columns
        else pd.Series([0.0] * len(dfx), index=dfx.index),    # 5
        range_break_frac,                 # 6
        is_big_normal_as0,                # 7
        is_big_maru_as0,                  # 8
        big_ratio_as0,                    # 9
        vol_spike_ratio,                  # 10
        dfx[COL_TIME].astype(str),        # 11
        idx_1h_col,                       # 12
        time_1h_col,                      # 13
    ))

    candle_hover = (
        "TF=15M<br>"
        "idx=%{customdata[0]}<br>"
        "time=%{customdata[11]}<br>"
        "idx_1H=%{customdata[12]}<br>"
        "time_1H=%{customdata[13]}<br>"
        "O=%{open}<br>"
        "H=%{high}<br>"
        "L=%{low}<br>"
        "C=%{close}<br>"
        "candle_type=%{customdata[3]}<br>"
        "body_len=%{customdata[4]:.5f}<br>"
        "candle_len=%{customdata[5]:.5f}<br>"
        "body_pct=%{customdata[2]:.2%}<br>"
        "mid_price=%{customdata[1]:.5f}<br>"
        "big_normal=%{customdata[7]}  big_maru=%{customdata[8]}  big_ratio=%{customdata[9]:.2f}<br>"
        "range_break_frac=%{customdata[6]:.2%}<br>"
        "vol_spike_ratio=%{customdata[10]:.2f}"
        "<extra></extra>"
    )

    # ===================================================================
    # Phase A: M15 base layer
    # ===================================================================

    # --- Candlesticks (with imbalance highlighting) ---
    imbalance_highlight = imbalance_cfg.get("highlight", False) and "is_imbalance" in dfx.columns

    if imbalance_highlight:
        is_imb = dfx["is_imbalance"] == 1
        direction = dfx["direction"]

        regular_mask = ~is_imb
        if regular_mask.any():
            regular_df = dfx[regular_mask]
            regular_cd = [customdata[i] for i in range(len(dfx)) if regular_mask.iloc[i]]
            fig.add_trace(go.Candlestick(
                x=regular_df[COL_TIME], open=regular_df[COL_O],
                high=regular_df[COL_H], low=regular_df[COL_L], close=regular_df[COL_C],
                name="OHLC", showlegend=False,
                customdata=regular_cd, hovertemplate=candle_hover,
            ))

        for imb_dir, imb_label, style_key in [
            (1, "BULLISH IMBALANCE", "imbalance.bullish"),
            (-1, "BEARISH IMBALANCE", "imbalance.bearish"),
        ]:
            mask = is_imb & (direction == imb_dir)
            if mask.any():
                sub_df = dfx[mask]
                sub_cd = [customdata[i] for i in range(len(dfx)) if mask.iloc[i]]
                color = _style(style_key).get("rgba", "rgba(128,128,128,0.8)")
                fig.add_trace(go.Candlestick(
                    x=sub_df[COL_TIME], open=sub_df[COL_O],
                    high=sub_df[COL_H], low=sub_df[COL_L], close=sub_df[COL_C],
                    name=f"imbalance:{imb_label.lower().split()[0]}",
                    showlegend=False,
                    increasing=dict(line=dict(color=color), fillcolor=color),
                    decreasing=dict(line=dict(color=color), fillcolor=color),
                    customdata=sub_cd,
                    hovertemplate=f"<b>{imb_label}</b><br>" + candle_hover,
                ))
    else:
        fig.add_trace(go.Candlestick(
            x=dfx[COL_TIME], open=dfx[COL_O],
            high=dfx[COL_H], low=dfx[COL_L], close=dfx[COL_C],
            name="OHLC", showlegend=False,
            customdata=customdata, hovertemplate=candle_hover,
        ))

    # --- Structure pattern markers (triangles) ---
    if "pat" in dfx.columns and "pat_dir" in dfx.columns and "pat_status" in dfx.columns:
        enabled_names = {k for k, v in pat_cfg.items() if v is True}
        sub = dfx[
            (dfx["pat"] != "")
            & (dfx["pat"].isin(enabled_names))
            & (dfx["pat_status"].isin([PatternStatus.SUCCESS.value, PatternStatus.CONFIRMED.value]))
        ]
        if not sub.empty:
            for status in [PatternStatus.SUCCESS.value, PatternStatus.CONFIRMED.value]:
                for d, pos_col, off_sign, style_sfx in [
                    (1, COL_H, 1, "up"),
                    (-1, COL_L, -1, "down"),
                ]:
                    pts = sub[(sub["pat_status"] == status) & (sub["pat_dir"] == d)]
                    if pts.empty:
                        continue
                    skey = f"structure.{'success' if status == PatternStatus.SUCCESS.value else 'confirmed'}.{style_sfx}"
                    fig.add_trace(go.Scatter(
                        x=pts[COL_TIME],
                        y=pts[pos_col] + off_sign * wick_offset.loc[pts.index],
                        mode="markers", name=f"struct:{status}:{d:+d}",
                        customdata=pts[["pat", "pat_dir", "pat_status",
                                        "pat_start_idx", "pat_end_idx", "pat_confirm_idx"]].values,
                        hovertemplate=(
                            "TF=15M<br>"
                            "pat=%{customdata[0]}<br>"
                            "dir=%{customdata[1]}<br>"
                            "status=%{customdata[2]}<br>"
                            "start=%{customdata[3]} end=%{customdata[4]} conf=%{customdata[5]}"
                            "<extra></extra>"
                        ),
                        **_style(skey),
                    ))

    # --- Volume spike markers ---
    if volume_cfg.get("spike_marker", False) and "is_vol_spike" in dfx.columns:
        spike_candles = dfx[dfx["is_vol_spike"] == True]
        if len(spike_candles) > 0:
            spike_y = (
                spike_candles["mid_price"] if "mid_price" in spike_candles.columns
                else (spike_candles[COL_H] + spike_candles[COL_L]) / 2.0
            )
            fig.add_trace(go.Scatter(
                x=spike_candles[COL_TIME], y=spike_y,
                mode="markers", name="volume:spike",
                marker=_style("volume.spike_marker").get("marker", {}),
                hoverinfo="skip",
            ))

    # --- Volume bars + EMA ---
    if volume_enabled:
        volume = dfx[COL_V].astype(float)
        vol_dir = _col_or_default("vol_dir", 0).astype(int)
        vol_ema20 = _col_or_default("vol_ema20", float("nan")).astype(float)

        colors = []
        for v_dir in vol_dir:
            if v_dir == 1:
                colors.append(_style("volume.bar.up").get("color", "rgba(0,180,0,0.7)"))
            elif v_dir == -1:
                colors.append(_style("volume.bar.down").get("color", "rgba(220,0,0,0.7)"))
            else:
                colors.append(_style("volume.bar.neutral").get("color", "rgba(128,128,128,0.7)"))

        vol_customdata = list(zip(dfx.index, dfx[COL_TIME].astype(str)))
        fig.add_trace(go.Bar(
            x=dfx[COL_TIME], y=volume, name="Volume",
            marker_color=colors, showlegend=False, yaxis="y2",
            customdata=vol_customdata,
            hovertemplate="TF=15M<br>idx=%{customdata[0]}<br>time=%{customdata[1]}<br>Volume: %{y:,.0f}<extra></extra>",
        ))

        if volume_cfg.get("ema_line", False):
            ema_style = _style("volume.ema_line").get("line", {"width": 1.5, "color": "rgba(0,100,255,0.8)"})
            fig.add_trace(go.Scatter(
                x=dfx[COL_TIME], y=vol_ema20,
                mode="lines", name="Vol EMA(20)", line=ema_style,
                showlegend=False, yaxis="y2",
                customdata=vol_customdata,
                hovertemplate="TF=15M<br>idx=%{customdata[0]}<br>time=%{customdata[1]}<br>EMA(20): %{y:,.0f}<extra></extra>",
            ))

    # ===================================================================
    # Phase B: M15 structure elements (per identity-tuple view of m15_df.attrs)
    # ===================================================================
    time_by_idx_m15 = {int(i): t for i, t in zip(dfx.index.to_numpy(), dfx[COL_TIME])}
    idx_set_m15 = set(map(int, dfx.index.to_numpy()))

    # Entity-wide aliases (idx is entity-absolute — no slice offset).
    lt_df = m15_df
    slice_begin = 0  # noqa: F841 — kept: entity-absolute idx, no offset
    lt_times = pd.to_datetime(lt_df[COL_TIME], utc=True)

    def _lt_time(idx: int):
        """Get M15 candle time at entity-absolute idx, or None if OOB."""
        if 0 <= idx < len(lt_times):
            return lt_times.iloc[idx]
        return None

    def _lt_full_idx(idx: int) -> int:
        """Identity — idx is entity-absolute already."""
        return idx

    # --- PRE-PASS: every sub's ownership fns + polylines, then the recency rule.
    # The recent-vs-prior rule (chart review 2026-09-22) compares the drawn
    # segments of DIFFERENT subs, so all of them must exist before any is drawn.
    sub_ctx: dict = {}
    _segments_by_sub: dict = {}
    _anchors_by_sub = {
        _sid_record_identity(r): int(r.creation_event_idx)
        for r in sid_records if r.creation_event_idx is not None
    }
    for _rec in sid_records:
        if m15_df.empty:
            continue
        _eid = _sid_record_identity(_rec)
        _is_live_fn, _owned_fn = _make_sub_phase_fns(
            _eid, int(_rec.starting_sd),
            int(_rec.start_idx) if _rec.start_idx is not None else None,
            live_by_idx_dir, forming_by_idx_dir,
        )
        _poly = None
        if struct_cfg.get("levels", False):
            _poly = _build_sub_polylines(
                _rec, events_by_sid.get(_eid, []), lt_df, _lt_time, _lt_full_idx, _owned_fn,
                sub_records=records_by_sub.get(_eid, []), anchors_by_sub=_anchors_by_sub,
            )
            _segments_by_sub[_eid] = (_recency_key(_rec), _segments_of(_poly))
        sub_ctx[_eid] = {"is_live": _is_live_fn, "owned_here": _owned_fn, "poly": _poly}
    prior_segs = _prior_line_segments(_segments_by_sub)

    for sid_rec in sid_records:
        # Each SidRecord (one unique sub) drives one rendering pass. Idx
        # values are entity-absolute.
        if m15_df.empty:
            continue

        eid = _sid_record_identity(sid_rec)
        sub_dir = int(sid_rec.starting_sd)
        sub_records = records_by_sub.get(eid, [])
        # Hover strings for this sub: its window + reason and its record list
        # `(lens, (S,C), trigger_type, trigger_idx→start_idx)` (§17.9).
        _win_end = sid_rec.end_event_idx if sid_rec.end_event_idx is not None else "open"
        sub_window_str = f"[{sid_rec.start_idx},{_win_end}] {sid_rec.end_reason or 'open'}"
        sub_records_str = "; ".join(
            f"{tr.lens}({tr.parent_sid},{tr.parent_cycle_id}) {tr.trigger_type} "
            f"{tr.trigger_idx}→{tr.start_idx}" + ("†" if tr.is_zero_length else "")
            for tr in sorted(sub_records, key=lambda r: (r.start_idx, r.seq))
        ) or "-"
        _segs = list(sid_rec.relative_dir_segments or ())

        def _relative_dir_at(idx: int, _segs=_segs) -> str:
            cur = "-"
            for from_idx, rd in _segs:
                if int(from_idx) <= int(idx):
                    cur = rd
                else:
                    break
            return cur
        sid_events = events_by_sid.get(eid, [])
        sid_kls = kls_by_sid.get(eid, [])
        sid_pois = pois_by_sid.get(eid, [])
        sid_fibs = fibs_by_sid.get(eid, [])  # noqa: F841 — fib lines off in default cfg
        sid_waves = waves_by_sid.get(eid, [])
        sid_wvmis = wvmis_by_sid.get(eid, [])
        sid_prev_bos = prev_bos_by_sid.get(eid, [])

        p_sid, p_cycle = _sid_parent(sid_rec)   # informational (first record)

        # §16.5 sid-tied filter (rev 2 + chart review 2026-09-20): a candle in
        # this sub's LIVE window is drawn iff this sub is its live owner for its
        # DIRECTION (later start wins among live subs); a candle in this sub's
        # FORMING span is drawn regardless of any live sub of the same direction
        # (a structure forming under a live one must stay visible), ownership
        # deciding only among forming subs (later anchor wins). Built in the
        # pre-pass (`_make_sub_phase_fns`) together with this sub's polylines.
        _sub_start = int(sid_rec.start_idx) if sid_rec.start_idx is not None else None
        _ctx = sub_ctx[eid]
        _is_live = _ctx["is_live"]
        _owned_here = _ctx["owned_here"]

        # Forming-phase ZONES are not drawn (chart review 2026-09-20, option 1):
        # a KL/POI zone whose cycle ended at/before the sub's `start_idx` was
        # collapsed by the sub's lifecycle floor (`status="inactive"`, empty
        # activation_history) — it existed geometrically but was never
        # tradeable. The forming dots/lines already show that geometry; the
        # zone rectangles only add clutter. The rows stay in the CSVs
        # (inspectable); only the rendering skips them. Shared predicate with
        # the H1 chart (`_zone_render.is_collapsed_cycle_zone`: clamped
        # confirmed_idx >= end_idx); POIs are keyed off their cycle's BOS KL
        # zone. A live-window POI that never activated (its cycle not
        # collapsed) is still drawn as an outline (pre-existing convention).
        _collapsed_sub_cycles = collapsed_cycles(sid_kls)

        # Determine if this sid's structure is the "most recent active"
        # by parent identifiers. Used downstream by `_render_m15_dots`
        # for opacity tiering of dot trace styles.
        last_m15_sid = (
            max((int(e.meta.get("structure_id", 0)) for e in sid_events), default=0)
            if sid_events else 0
        )
        is_active_trigger = (
            p_sid == m15_most_recent_psid and p_cycle in m15_recent_cycles
        )

        # --- Structure swing lines ---
        if struct_cfg.get("levels", False) and _ctx["poly"] is not None:
            _poly = _ctx["poly"]
            points_by_sid = _poly["points_by_sid"]
            extra_cts_pts = _poly["extra_cts_pts"]
            extra_pb_pts = _poly["extra_pb_pts"]
            pb_to_bos_lines = _poly["pb_to_bos_lines"]
            most_recent_lt_sid = _poly["most_recent_lt_sid"]

            # RECENT-vs-PRIOR rule (chart review 2026-09-22): a segment is drawn
            # in the dotted PRIOR style iff a structure with a higher
            # `_recency_key` draws a segment over the same candles (more than
            # one shared candle, any direction) — `prior_segs`, computed once
            # per lens in the pre-pass. Everything else is solid, including
            # structures that were never live in real time (nothing supersedes
            # them there). Consecutive segments sharing a style are drawn as one
            # run (`_group_flag_runs`), the boundary point belonging to both.
            # The endpoints of solid segments go into `recent_pts` so the dots
            # follow their segments; the hover `phase` stays the real-time fact.
            recent_pts: set = set()   # {(internal sid, idx)}: ends a solid segment
            for sid, seq in sorted(_poly["seq_by_sid"].items()):
                if len(seq) == 1:
                    # A lone point has no segment to decide it — draw it solid.
                    recent_pts.add((seq[0][4], seq[0][0]))
                    continue
                flags = [(eid, ("w", sid, i)) not in prior_segs for i in range(len(seq) - 1)]
                for is_recent, i0, i1 in _group_flag_runs(flags):
                    run = seq[i0:i1 + 1]
                    if is_recent:
                        recent_pts.update((p[4], p[0]) for p in run if p[3] != "EXT")
                        line_style = _style("structure.m15.swing_line").copy()
                        trace_name = f"M15 swing sub{eid}_m15s{sid}"
                    else:
                        line_style = _style("structure.m15.swing_line_prior").copy()
                        trace_name = f"M15 swing (prior) sub{eid}_m15s{sid}"
                    fig.add_trace(go.Scatter(
                        x=[p[1] for p in run], y=[p[2] for p in run], mode="lines",
                        name=trace_name, hoverinfo="skip", line_shape="linear",
                        showlegend=False, **line_style,
                    ))

            # Cross-structure PB→BOS lines take part in the same rule.
            for _k, (_pb_sid, pb_t, pb_p, bos_t, bos_p, pb_idx, bos_idx) in enumerate(pb_to_bos_lines):
                is_recent = (eid, ("pb", _k)) not in prior_segs
                if is_recent:
                    recent_pts.add((_pb_sid, pb_idx))
                    recent_pts.add((_pb_sid + 1, bos_idx))
                line_style = _style(
                    "structure.m15.swing_line" if is_recent else "structure.m15.swing_line_prior"
                ).copy()
                fig.add_trace(go.Scatter(
                    x=[pb_t, bos_t], y=[pb_p, bos_p], mode="lines",
                    name=f"M15 PB→BOS sub{eid}",
                    hoverinfo="skip", line_shape="linear", showlegend=False, **line_style,
                ))

            # --- CTS confirmed dots ---
            all_cts_pts = []
            for sid_pts in points_by_sid.values():
                for p in sid_pts:
                    if p[3] == "CTS":
                        all_cts_pts.append(p)
            if all_cts_pts:
                _render_m15_dots(fig, all_cts_pts, "CTS", p_sid, p_cycle, eid,
                                 m15_most_recent_psid, m15_recent_cycles, is_active_trigger,
                                 most_recent_lt_sid, m15_to_h1,
                                 sub_window_str, sub_records_str, _relative_dir_at, _is_live,
                                 recent_pts)

            # --- BOS confirmed dots ---
            all_bos_pts = []
            for sid_pts in points_by_sid.values():
                for p in sid_pts:
                    if p[3] == "BOS":
                        all_bos_pts.append(p)
            if all_bos_pts:
                _render_m15_dots(fig, all_bos_pts, "BOS", p_sid, p_cycle, eid,
                                 m15_most_recent_psid, m15_recent_cycles, is_active_trigger,
                                 most_recent_lt_sid, m15_to_h1,
                                 sub_window_str, sub_records_str, _relative_dir_at, _is_live,
                                 recent_pts)

            # --- Unconfirmed CTS dots ---
            if extra_cts_pts:
                _render_m15_dots(fig, extra_cts_pts, "CTS (unconf)", p_sid, p_cycle, eid,
                                 m15_most_recent_psid, m15_recent_cycles, is_active_trigger,
                                 most_recent_lt_sid, m15_to_h1,
                                 sub_window_str, sub_records_str, _relative_dir_at, _is_live,
                                 recent_pts)

            # --- PB dots ---
            if extra_pb_pts:
                _render_m15_dots(fig, extra_pb_pts, "PB", p_sid, p_cycle, eid,
                                 m15_most_recent_psid, m15_recent_cycles, is_active_trigger,
                                 most_recent_lt_sid, m15_to_h1,
                                 sub_window_str, sub_records_str, _relative_dir_at, _is_live,
                                 recent_pts)

        # --- KL zone rectangles + hover ---
        if zone_cfg.get("KL", False) and sid_kls:
            t_last_m15 = dfx[COL_TIME].iloc[-1]

            for zone in sid_kls:
                if is_collapsed_cycle_zone(zone):
                    continue
                side = str(zone.side)
                stz = _zone_style(side)
                op_mult = _m15_opacity_tier_for_zone(zone)

                base_fill_op = float(stz.get("fill_opacity_active", 0.4))
                base_line_op = float(stz.get("confirm_opacity_active", 0.9))
                fill_op = base_fill_op * op_mult
                line_op = base_line_op * op_mult
                rgb = str(stz.get("rgb", "0,180,0" if side == "buy" else "220,0,0"))
                confirm_w = int(stz.get("confirm_line_width", 2))

                fillcolor = _rgba_from_rgb(rgb, fill_op)
                confirm_color = _rgba_from_rgb(rgb, line_op)
                # Sub-native outline keeps the thin black 0.5px styling, opacity
                # tracks the line opacity for tier-consistent dimming.
                sub_outline_color = f"rgba(0, 0, 0, {line_op})"
                sub_outline_w = 0.5

                x0 = pd.to_datetime(zone.start_time, utc=True)
                x1 = pd.to_datetime(zone.end_time, utc=True) if zone.end_time else t_last_m15

                y0 = float(min(zone.top, zone.bottom))
                y1 = float(max(zone.top, zone.bottom))

                conf_idx = int(zone.meta.get("confirmed_idx", -1))
                conf_time = None
                # conf_idx is entity-absolute under §13.5.c.iii
                if conf_idx in lt_df.index:
                    conf_time = pd.to_datetime(lt_df.loc[conf_idx, COL_TIME], utc=True)

                steps = list((zone.meta or {}).get("bounds_steps", []))
                if not steps:
                    steps = [{"start_idx": int(zone.meta.get("base_idx", 0)),
                              "top": y1, "bottom": y0, "event": "FALLBACK"}]
                steps = sorted(steps, key=lambda s: int(s.get("start_idx", -1)))

                # render_end_idx for active-stretch computation.
                if zone.end_time is None:
                    render_end_idx = int(lt_df.index[-1])
                else:
                    _em = lt_df[COL_TIME] <= pd.to_datetime(zone.end_time, utc=True)
                    render_end_idx = int(lt_df.index[_em][-1]) if _em.any() else int(lt_df.index[0])

                active_stretches = compute_kl_active_stretches(zone, render_end_idx)

                # Hover identity = the sub's `sub_id`. The zone meta carries the
                # *internal* MS structure_id from the bounded sub run, which
                # restarts at 0 per sub — useless for telling subs apart.
                sub_id = eid
                cycle_id = int(zone.meta.get("cycle_id", 0))

                # Per-step iteration: fills (only where active) + hover lines.
                for k, s in enumerate(steps):
                    seg_start_slice = int(s.get("start_idx", -1))
                    seg_x0 = _lt_time(seg_start_slice)
                    if seg_x0 is None:
                        continue
                    if seg_x0 < x0:
                        seg_x0 = x0

                    if k + 1 < len(steps):
                        next_step_idx = int(steps[k + 1].get("start_idx", -1))
                        nxt = _lt_time(next_step_idx) or x1
                        seg_x1 = nxt - pd.Timedelta(microseconds=1)
                        step_end_idx = next_step_idx - 1
                    else:
                        seg_x1 = x1
                        step_end_idx = render_end_idx

                    seg_top = float(s.get("top", y1))
                    seg_bot = float(s.get("bottom", y0))
                    sy0 = min(seg_bot, seg_top)
                    sy1 = max(seg_bot, seg_top)

                    for stretch_start, stretch_end in active_stretches:
                        isect_start = max(seg_start_slice, stretch_start)
                        isect_end = min(step_end_idx, stretch_end)
                        if isect_start > isect_end:
                            continue
                        fill_x0 = _lt_time(isect_start) or seg_x0
                        if fill_x0 < seg_x0:
                            fill_x0 = seg_x0
                        if isect_end >= step_end_idx:
                            fill_x1 = seg_x1
                        else:
                            end_time = _lt_time(isect_end)
                            fill_x1 = end_time if end_time is not None else seg_x1
                        if fill_x1 <= fill_x0:
                            continue
                        fig.add_shape(
                            type="rect", xref="x", yref="y",
                            x0=fill_x0, x1=fill_x1, y0=sy0, y1=sy1,
                            fillcolor=fillcolor,
                            line=dict(width=0),
                            layer="below",
                        )

                    # Hover lines per step (preserved)
                    seg_times = dfx[COL_TIME][(dfx[COL_TIME] >= seg_x0) & (dfx[COL_TIME] <= seg_x1)]
                    if len(seg_times) == 0:
                        seg_times = pd.Series([seg_x0, seg_x1])

                    hover_cd = [[
                        side, sub_id,
                        int(zone.meta.get("struct_direction", 0)),
                        str(zone.meta.get("base_pattern", "")),
                        int(zone.meta.get("base_idx", -1)),
                        conf_idx, cycle_id,
                        sy1, sy0, p_sid, p_cycle,
                        sub_window_str, sub_records_str,
                    ]] * len(seg_times)

                    kl_hover_line = {"width": 6, "color": "rgba(0,0,0,0)"}
                    for yval in (sy1, sy0):
                        fig.add_trace(go.Scatter(
                            x=seg_times, y=[yval] * len(seg_times),
                            mode="lines", name="M15 KL zone", showlegend=False,
                            line=kl_hover_line, line_shape="hv",
                            hovertemplate=(
                                "TF=15M<br>"
                                "KL Zone<br>"
                                "side=%{customdata[0]}<br>"
                                "sub_id=%{customdata[1]} | first record parent_sid=%{customdata[9]} parent_cycle_id=%{customdata[10]}<br>"
                                "sub window=%{customdata[11]}<br>"
                                "records: %{customdata[12]}<br>"
                                "cycle_id=%{customdata[6]}<br>"
                                "struct_direction=%{customdata[2]}<br>"
                                "base_pattern=%{customdata[3]}<br>"
                                "base_idx=%{customdata[4]}<br>"
                                "confirmed_idx=%{customdata[5]}<br>"
                                "top=%{customdata[7]:.5f}<br>"
                                "bottom=%{customdata[8]:.5f}"
                                "<extra></extra>"
                            ),
                            customdata=hover_cd,
                        ))

                # Stepped polygon outline (single trace for whole zone).
                outline_xs, outline_ys = build_stepped_outline_xy(
                    steps, _lt_time, x0, x1,
                    fallback_top=y1, fallback_bottom=y0,
                )
                fig.add_trace(go.Scatter(
                    x=outline_xs, y=outline_ys, mode="lines",
                    line=dict(color=sub_outline_color, width=sub_outline_w),
                    fill=None, hoverinfo="skip", showlegend=False,
                    name=f"M15 KL outline sub{sub_id} c{cycle_id}",
                ))

                # Single confirm line at confirmed_idx (KL has no reactivation).
                if conf_time is not None and x0 <= conf_time <= x1:
                    for k, s in enumerate(steps):
                        seg_x0 = _lt_time(int(s.get("start_idx", -1)))
                        if seg_x0 is None:
                            continue
                        if seg_x0 < x0:
                            seg_x0 = x0
                        if k + 1 < len(steps):
                            nxt = _lt_time(int(steps[k + 1].get("start_idx", -1))) or x1
                            seg_x1 = nxt - pd.Timedelta(microseconds=1)
                        else:
                            seg_x1 = x1
                        if seg_x0 <= conf_time <= seg_x1:
                            seg_top = float(s.get("top", y1))
                            seg_bot = float(s.get("bottom", y0))
                            fig.add_shape(
                                type="line", xref="x", yref="y",
                                x0=conf_time, x1=conf_time,
                                y0=min(seg_bot, seg_top), y1=max(seg_bot, seg_top),
                                line=dict(color=confirm_color, width=confirm_w),
                                layer="below",
                            )
                            break

        # --- Wave candle verticals (per-candle lifecycle gating) ---
        # WAVE_CANDLES_SPEC "Chart Rendering & Lifecycle": each line gated by
        # its cycle's per-candle lifecycle (FB=cycle_start; LB/FP/LP=CTS_conf
        # clamped inside compute_wave_candle_visibility; all end at cycle end).
        # BOS.last cross-cycles to cycle N−1 (locked LP from prev cycle); LP
        # additionally hidden when ended without lock. `BOS_0.last` (Option C):
        # render iff sub's cyc=0 is non-collapsed. Overlap kept via `_owned_here`
        # per-candle owner filter (§16.5 sid-tied filter, same as the rest of
        # the sub block). Uniform opacity; wave-candle hover only (no WVMI
        # momentum / weighted vol).
        if zone_cfg.get("wave_candles", True) and sid_waves:
            from engine_v2.zones.wave_candles import (
                _role_for_wave_candle,
                compute_wave_candle_visibility,
            )

            # cycle_life per (internal sid, cycle) from this sub's KL BOS zones.
            # KL meta already encodes (cycle_start=confirmed_idx, end_idx,
            # end_reason) computed with the slice-local floor/cap at build time.
            cycle_life: dict = {}
            for z in sid_kls:
                if z.source_kind != "BOS":
                    continue
                m = z.meta or {}
                s, c = int(m.get("structure_id", 0)), int(m.get("cycle_id", 0))
                cycle_life[(s, c)] = (
                    m.get("confirmed_idx"),
                    m.get("end_idx"),
                    m.get("end_reason"),
                )
            cts_conf: dict = {}
            for ev in sid_events:
                if getattr(ev, "type", None) != "CTS_CONFIRMED":
                    continue
                em = ev.meta or {}
                s = em.get("structure_id"); c = em.get("cycle_id")
                if s is None or c is None:
                    continue
                cts_conf[(int(s), int(c))] = int(ev.idx)
            wc_visibility = compute_wave_candle_visibility(cycle_life, cts_conf)

            wc_y_min = float(dfx[COL_L].min())
            wc_y_max = float(dfx[COL_H].max())

            for wc in sid_waves:
                for position, idx in (("last", wc.last_wave_candle_idx),
                                      ("first", wc.first_wave_candle_idx)):
                    if idx is None or idx >= len(lt_df):
                        continue
                    # §16.5 sid-tied filter: only the owning sid renders at
                    # each candle (same as the rest of the sub block).
                    if not _owned_here(idx):
                        continue
                    role_info = _role_for_wave_candle(str(wc.source_kind), position)
                    if role_info is None:
                        continue
                    role, cycle_offset = role_info
                    lookup_cycle = wc.cycle_id + cycle_offset
                    if lookup_cycle < 0:
                        # BOS_0.last (pre-structure pullback). Option C
                        # (2026-05-28): render iff this sub's cyc=0 is
                        # non-collapsed — clean structure start shows the
                        # pre-structure pullback; collapsed cyc=0
                        # (retroactive-Scenario-2 phantom) hides it.
                        life0 = cycle_life.get((wc.structure_id, 0))
                        if life0 is None:
                            continue
                        cstart0, cend0, _r0 = life0
                        if cstart0 is None or (cend0 is not None and cstart0 >= cend0):
                            continue
                        # cyc=0 non-collapsed → render.
                    else:
                        viz = wc_visibility.get((wc.structure_id, lookup_cycle, role))
                        if viz is None or not viz[0]:
                            continue
                    candle_dir = int(lt_df.iloc[idx]["direction"])
                    if candle_dir == 0:
                        continue

                    style_key = "wave_candle.bullish" if candle_dir == 1 else "wave_candle.bearish"
                    wc_style = _style(style_key)
                    line_info = wc_style.get("line", {})
                    color_rgb = line_info.get("color_rgb", "128,128,128")
                    base_opacity = float(wc_style.get("opacity", 0.8))
                    line_width = int(line_info.get("width", 1))
                    dash = line_info.get("dash")

                    # Uniform opacity (no tier multiplier).
                    final_color = f"rgba({color_rgb}, {base_opacity})"
                    wc_time = _lt_time(idx)
                    if wc_time is None:
                        continue

                    line_props = dict(color=final_color, width=line_width)
                    if dash:
                        line_props["dash"] = dash
                    fig.add_shape(type="line", xref="x", yref="paper",
                                  x0=wc_time, x1=wc_time, y0=0, y1=1,
                                  line=line_props, layer="below")

                    # Wave-candle hover (no WVMI momentum / Weighted vol).
                    vol = float(lt_df.iloc[idx]["volume"])
                    full_idx = _lt_full_idx(idx)
                    h1_info = m15_to_h1.get(wc_time, (None, None))
                    hover_lines = [
                        "TF=15M",
                        "<b>Wave Candle</b>",
                        f"idx={full_idx}  idx_1H={h1_info[1]}",
                        # TODO: label hardcodes "BOS zone:" but CTS wave candles render here too
                        # (subs include both BOS+CTS source_kinds internally — orchestrator §5).
                        # `wc.structure_id` is the bounded sub's internal MS sid (restarts at 0
                        # per sub) — use sub_id for the user-facing identity instead.
                        f"BOS zone: sub_id={eid} cycle={wc.cycle_id}",
                        f"first record parent_sid={p_sid} parent_cycle={p_cycle}",
                        f"sub window={sub_window_str}",
                        f"Volume: {vol:.0f}",
                    ]

                    _n_pts = 12
                    _y_pts = [wc_y_min + i * (wc_y_max - wc_y_min) / (_n_pts - 1) for i in range(_n_pts)]
                    fig.add_trace(go.Scatter(
                        x=[wc_time] * _n_pts, y=_y_pts,
                        mode="lines", showlegend=False,
                        line=dict(width=8, color="rgba(0,0,0,0)"),
                        hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
                    ))

        # --- POI zone rectangles + hover ---
        if zone_cfg.get("POI", False) and sid_pois:
            t_last_m15 = dfx[COL_TIME].iloc[-1]

            for poi in sid_pois:
                if poi.meta.get("status") == "disappeared":
                    continue
                if is_poi_of_collapsed_cycle(poi, _collapsed_sub_cycles):
                    continue
                side = str(poi.side)
                stz = STYLE.get(f"zone.poi.{side}", {})
                zone_status = poi.meta.get("status", "active")

                op_mult = _m15_opacity_tier_for_zone(poi)

                base_fill_op = float(stz.get("fill_opacity_active", 0.9))
                base_line_op = float(stz.get("confirm_opacity_active", 0.9))
                rgb = str(stz.get("rgb", "255, 215, 0"))
                confirm_rgb = str(stz.get("confirm_line_rgb", "101, 67, 33"))
                confirm_w = int(stz.get("confirm_line_width", 2))

                fill_op = base_fill_op * op_mult
                line_op = base_line_op * op_mult
                fillcolor = _rgba_from_rgb(rgb, fill_op)
                confirm_color = _rgba_from_rgb(confirm_rgb, line_op)
                # Sub-native outline keeps thin black 0.5px (tier-faded).
                sub_outline_color = f"rgba(0, 0, 0, {line_op})"
                sub_outline_w = 0.5

                x0 = pd.to_datetime(poi.start_time, utc=True)
                x1 = pd.to_datetime(poi.end_time, utc=True) if poi.end_time else t_last_m15

                y0 = float(min(poi.top, poi.bottom))
                y1 = float(max(poi.top, poi.bottom))

                # `confirmed_idx` can be None when the POI never activated within
                # its lifetime (per lifecycle convention).
                _conf_raw = poi.meta.get("confirmed_idx")
                conf_idx = int(_conf_raw) if _conf_raw is not None else None

                # render_end_idx for active-stretch computation.
                if poi.end_time is None:
                    render_end_idx = int(lt_df.index[-1])
                else:
                    _em = lt_df[COL_TIME] <= pd.to_datetime(poi.end_time, utc=True)
                    render_end_idx = int(lt_df.index[_em][-1]) if _em.any() else int(lt_df.index[0])

                poi_stretches = compute_poi_active_stretches(poi, render_end_idx)

                # Active-stretch fills.
                for stretch_start, stretch_end in poi_stretches:
                    if stretch_start not in lt_df.index or stretch_end not in lt_df.index:
                        # Fall back to nearest available idx within range.
                        s_clip = max(int(lt_df.index[0]), min(stretch_start, int(lt_df.index[-1])))
                        e_clip = max(int(lt_df.index[0]), min(stretch_end, int(lt_df.index[-1])))
                        if s_clip not in lt_df.index or e_clip not in lt_df.index:
                            continue
                        stretch_start, stretch_end = s_clip, e_clip
                    sx0 = pd.to_datetime(lt_df.loc[stretch_start, COL_TIME], utc=True)
                    sx1 = pd.to_datetime(lt_df.loc[stretch_end, COL_TIME], utc=True)
                    if sx0 < x0:
                        sx0 = x0
                    if sx1 > x1:
                        sx1 = x1
                    if sx1 <= sx0:
                        continue
                    fig.add_shape(
                        type="rect", xref="x", yref="y",
                        x0=sx0, x1=sx1, y0=y0, y1=y1,
                        fillcolor=fillcolor,
                        line=dict(width=0),
                        layer="below",
                    )

                # Outline rect (no fill).
                fig.add_shape(
                    type="rect", xref="x", yref="y",
                    x0=x0, x1=x1, y0=y0, y1=y1,
                    fillcolor="rgba(0,0,0,0)",
                    line=dict(color=sub_outline_color, width=sub_outline_w),
                    layer="below",
                )

                # N vertical confirm lines — one per "A" event in activation_history
                # (falls back to single line at confirmed_idx for legacy zones).
                activation_history = poi.meta.get("activation_history", []) or []
                confirm_idxs = [
                    int(ev["idx"]) for ev in activation_history
                    if ev.get("active") and int(ev["idx"]) <= render_end_idx
                ]
                if not confirm_idxs and conf_idx is not None:
                    confirm_idxs = [conf_idx]
                for c_idx in confirm_idxs:
                    if c_idx not in lt_df.index:
                        continue
                    c_time = pd.to_datetime(lt_df.loc[c_idx, COL_TIME], utc=True)
                    if not (x0 <= c_time <= x1):
                        continue
                    fig.add_shape(
                        type="line", xref="x", yref="y",
                        x0=c_time, x1=c_time, y0=y0, y1=y1,
                        line=dict(color=confirm_color, width=confirm_w),
                        layer="below",
                    )

                seg_times = dfx[COL_TIME][(dfx[COL_TIME] >= x0) & (dfx[COL_TIME] <= x1)]
                if len(seg_times) == 0:
                    seg_times = pd.Series([x0, x1])

                versions = poi.meta.get("versions", [])
                versions_str = ", ".join(versions) if versions else "none"
                # `poi.meta["structure_id"]` is the bounded sub's internal MS
                # sid (restarts at 0 per sub) — use sub_id for user-facing identity.
                poi_cd = [[
                    side, eid, poi.meta.get("struct_direction", 0),
                    poi.ic_idx, conf_idx, poi.meta.get("cycle_id", 0),
                    versions_str, y1, y0, zone_status, p_sid, p_cycle,
                    sub_window_str, sub_records_str,
                ]] * len(seg_times)

                poi_hover_line = {"width": 6, "color": "rgba(0,0,0,0)"}
                for yval in (y1, y0):
                    fig.add_trace(go.Scatter(
                        x=seg_times, y=[yval] * len(seg_times),
                        mode="lines", name="M15 POI zone", showlegend=False,
                        line=poi_hover_line, line_shape="hv",
                        hovertemplate=(
                            "TF=15M<br>"
                            "<b>POI Zone</b><br>"
                            "side=%{customdata[0]}<br>"
                            "sub_id=%{customdata[1]} | first record parent_sid=%{customdata[10]} parent_cycle_id=%{customdata[11]}<br>"
                            "sub window=%{customdata[12]}<br>"
                            "records: %{customdata[13]}<br>"
                            "cycle_id=%{customdata[5]}<br>"
                            "struct_direction=%{customdata[2]}<br>"
                            "ic_idx=%{customdata[3]}<br>"
                            "confirmed_idx=%{customdata[4]}<br>"
                            "versions=%{customdata[6]}<br>"
                            "top=%{customdata[7]:.5f}<br>"
                            "bottom=%{customdata[8]:.5f}<br>"
                            "status=%{customdata[9]}"
                            "<extra></extra>"
                        ),
                        customdata=poi_cd,
                    ))

        # --- Prev BOS lines ---
        for line_info in sid_prev_bos:
            start_slice = line_info["start_idx"]
            end_slice = line_info["end_idx"]
            price = line_info["price"]
            # §16.5 sid-tied filter: the owning sub at the line's start candle
            # draws it. Prev-BOS lines carry NO lifecycle formatting (chart
            # review 2026-09-21): always solid, on the sub charts and on main.
            if not _owned_here(start_slice):
                continue
            t0 = _lt_time(start_slice)
            t1 = _lt_time(end_slice)
            if t0 is not None and t1 is not None:
                op_mult = _m15_opacity_tier_for_events(
                    p_sid, p_cycle, m15_most_recent_psid, m15_recent_cycles, False,
                )
                line_s = _style("prev_bos_line.m15").get("line", {"width": 2, "color": "royalblue"})
                fig.add_trace(go.Scatter(
                    x=[t0, t1], y=[price, price], mode="lines", line=line_s,
                    name=f"M15 Prev BOS h1s{p_sid}c{p_cycle}", showlegend=False,
                    hoverlabel=dict(bgcolor="royalblue", font_color="white"),
                    hovertemplate=(
                        f"TF=15M<br>"
                        f"Prev BOS Line<br>"
                        f"Price: {price:.5f}<br>"
                        f"sub_id={eid} | first record parent_sid={p_sid} parent_cycle={p_cycle}<br>"
                        f"<extra></extra>"
                    ),
                ))

    # ===================================================================
    # Phase C: H1 overlay
    # ===================================================================
    _render_h1_overlay(fig, dfx, h1_df, h1_to_m15, m15_to_h1,
                       state_cfg, struct_cfg, zone_cfg)

    # Parent (H1) zone-proximity triggers, anchored at the matching M15
    # sub-candle. Parent-event overlay, so no §16.5 owner filter applies.
    _render_proximity_triggers_overlay(fig, dfx, h1_df, wick_offset)

    # ===================================================================
    # Phase D: Layout + Output
    # ===================================================================
    fig.update_layout(
        title=title,
        xaxis_title="Time (UTC)" if not volume_enabled else None,
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        legend_title="Overlays",
        height=800,
    )

    if volume_enabled:
        fig.update_layout(
            xaxis=dict(anchor="y2", side="bottom", title_text="Time (UTC)", showline=False),
            yaxis=dict(domain=[0.15, 1.0], showline=False),
            yaxis2=dict(domain=[0, 0.15], showgrid=False, zeroline=False,
                        showticklabels=True, side="right", tickformat=",", showline=False),
        )
        border_color = "rgba(0,0,0,0.4)"
        for x0, x1, y0, y1 in [(0, 1, 1, 1), (0, 1, 0, 0), (0, 0, 0, 1), (1, 1, 0, 1)]:
            fig.add_shape(type="line", xref="paper", yref="paper",
                          x0=x0, x1=x1, y0=y0, y1=y1,
                          line=dict(color=border_color, width=1))

    fig.update_layout(**_style("chart.layout"))
    fig.update_xaxes(**_style("chart.axis"))
    fig.update_yaxes(**_style("chart.axis"))

    # Dynamic gap removal
    t = pd.to_datetime(dfx[COL_TIME]).sort_values().reset_index(drop=True)
    dt = t.diff()
    expected = dt[dt.notna()].median()
    if pd.isna(expected) or expected <= pd.Timedelta(0):
        expected = pd.Timedelta(minutes=15)

    gap_mask = dt > (expected * 2.0)
    missing = []
    t_values = t.to_list()
    for i in range(1, len(t_values)):
        if bool(gap_mask.iloc[i]):
            start = t_values[i - 1] + expected
            end = t_values[i] - expected
            if start <= end:
                missing.extend(pd.date_range(start, end, freq=expected).to_pydatetime())

    present = set(t_values)
    missing = [x for x in missing if x not in present]
    if missing:
        fig.update_xaxes(
            rangebreaks=[dict(values=missing, dvalue=int(expected / pd.Timedelta(milliseconds=1)))]
        )

    html_path = out_dir / f"{basename}.html"
    png_path = out_dir / f"{basename}.png"

    print(f"[m15_chart] traces: {len(fig.data)}, shapes: {len(fig.layout.shapes) if fig.layout.shapes else 0}")

    fig.write_html(str(html_path), include_plotlyjs="cdn")

    # Volume auto-rescale JS
    if volume_enabled:
        _inject_volume_autoscale_js(html_path)

    fig.write_image(str(png_path), scale=2)

    return ChartExportPaths(html_path=html_path, png_path=png_path)


# ---------------------------------------------------------------------------
# Helper: Render M15 structure dots (CTS/BOS/PB)
# ---------------------------------------------------------------------------

def _render_m15_dots(
    fig, pts, kind_label, p_sid, p_cycle, sub_id,
    most_recent_psid, recent_cycles, is_active_trigger,
    most_recent_lt_sid, m15_to_h1,
    sub_window_str="", sub_records_str="", relative_dir_at=None, is_live=None,
    recent_pts=None,
):
    """Render M15 structure dots with TF=15M hover.

    `sub_id` is the unique sub's identity (§17.9). The points' `p[4]` is the
    bounded sub's *internal* MS structure_id which restarts at 0 per sub —
    useless for telling subs apart in the hover. `sub_window_str` /
    `sub_records_str` carry the sub's lifecycle window + reason and its
    record list; `relative_dir_at(idx)` gives the §17.3 step function at the
    dot's candle. Dots follow their SEGMENTS (chart review 2026-09-22): a point
    in `recent_pts` (`{(internal sid, idx)}` — the endpoints of the solid
    segments, and any point no segment decides) is drawn filled in the RECENT
    style, every other point as a dimmed open PRIOR marker — two traces.
    `is_live(idx)` is the real-time fact (`idx >= start_idx`) and only feeds the
    hover `phase`; without `recent_pts` it decides the style too (legacy
    per-point split).
    """
    if is_live is not None:
        def _styled_recent(p) -> bool:
            if recent_pts is not None:
                return (p[4], p[0]) in recent_pts
            return bool(is_live(p[0]))
        recent = [p for p in pts if _styled_recent(p)]
        prior = [p for p in pts if not _styled_recent(p)]
        if prior:
            _render_m15_dots_layer(
                fig, prior, kind_label, p_sid, p_cycle, sub_id, m15_to_h1,
                sub_window_str, sub_records_str, relative_dir_at, layer="prior",
                phase_at=is_live,
            )
        if recent:
            _render_m15_dots_layer(
                fig, recent, kind_label, p_sid, p_cycle, sub_id, m15_to_h1,
                sub_window_str, sub_records_str, relative_dir_at, layer="recent",
                phase_at=is_live,
            )
        return
    _render_m15_dots_layer(
        fig, pts, kind_label, p_sid, p_cycle, sub_id, m15_to_h1,
        sub_window_str, sub_records_str, relative_dir_at, layer="recent",
    )


def _render_m15_dots_layer(
    fig, pts, kind_label, p_sid, p_cycle, sub_id, m15_to_h1,
    sub_window_str, sub_records_str, relative_dir_at, layer, phase_at=None,
):
    """`layer` picks the marker STYLE (`recent` = filled, `prior` = dimmed open
    circle) and is reported in the hover; the hover `phase` per point is
    `phase_at(idx)` — the real-time fact — when given, so a filled dot that was
    never live still hovers `phase=forming` and a dotted-segment dot that WAS
    live still hovers `phase=live`."""
    base = "structure.m15.cts" if "CTS" in kind_label else "structure.m15.bos"
    style_key = base if layer == "recent" else base + "_prior"
    style = _style(style_key).copy()

    cd = []
    for p in pts:
        # p = (slice_idx, time, price, kind, sid, cycle, sd, full_idx)
        h1_info = m15_to_h1.get(p[1], (None, None))
        cd.append([
            p[7],  # full M15 idx
            p[3],  # kind
            p[2],  # price
            sub_id,  # unique sub identity (not p[4])
            p[5],  # m15 cycle_id
            p[6],  # sd
            p_sid,  # first record parent_sid (informational)
            p_cycle,  # first record parent_cycle_id (informational)
            h1_info[1] if h1_info[1] is not None else "",  # idx_1H
            str(h1_info[0]) if h1_info[0] is not None else "",  # time_1H
            relative_dir_at(p[7]) if relative_dir_at is not None else "",  # relative_dir at this candle
            sub_window_str,   # [start_idx,end_idx] reason
            sub_records_str,  # record list
            ("live" if phase_at(p[0]) else "forming") if phase_at is not None else "live",  # real-time fact
            layer,            # recent | prior (why this style — see _prior_line_segments)
        ])

    fig.add_trace(go.Scatter(
        x=[p[1] for p in pts],
        y=[p[2] for p in pts],
        mode="markers",
        name=f"M15 {kind_label} sub{sub_id}" + (" (prior)" if layer == "prior" else ""),
        showlegend=False,
        customdata=cd,
        hoverlabel=dict(bgcolor="royalblue", font_color="white"),
        hovertemplate=(
            "TF=15M<br>"
            "idx=%{customdata[0]}<br>"
            "idx_1H=%{customdata[8]}<br>"
            "kind=%{customdata[1]}<br>"
            "price=%{customdata[2]:.5f}<br>"
            "sub_id=%{customdata[3]} | first record parent_sid=%{customdata[6]} parent_cycle_id=%{customdata[7]}<br>"
            "phase=%{customdata[13]} | layer=%{customdata[14]}<br>"
            "relative_dir=%{customdata[10]}<br>"
            "sub window=%{customdata[11]}<br>"
            "records: %{customdata[12]}<br>"
            "cycle_id=%{customdata[4]}<br>"
            "struct_direction=%{customdata[5]}"
            "<extra></extra>"
        ),
        **style,
    ))


# ---------------------------------------------------------------------------
# Phase C: H1 overlay rendering
# ---------------------------------------------------------------------------

def _render_h1_overlay(fig, dfx, h1_df, h1_to_m15, m15_to_h1, state_cfg, struct_cfg, zone_cfg):
    """Render all H1 elements as overlays on the M15 chart."""
    h1_times = pd.to_datetime(h1_df[COL_TIME], utc=True)
    structure_events = h1_df.attrs.get("structure_events", [])
    m15_t_last = dfx[COL_TIME].iloc[-1]
    m15_t_first = dfx[COL_TIME].iloc[0]

    def _h1_to_m15_time(h1_time):
        """Map H1 time to M15 4th candle, clamped to M15 range."""
        t = h1_to_m15.get(h1_time, h1_time + timedelta(minutes=45))
        if t < m15_t_first:
            return m15_t_first
        if t > m15_t_last:
            return m15_t_last
        return t

    def _h1_idx_to_m15_time(h1_idx):
        """Map H1 index to M15 4th candle time."""
        if h1_idx not in h1_df.index:
            return None
        h1_time = pd.to_datetime(h1_df.loc[h1_idx, COL_TIME], utc=True)
        return _h1_to_m15_time(h1_time)

    # Find most recent H1 sid for opacity
    all_h1_sids = set()
    for ev in structure_events:
        sid = ev.meta.get("structure_id")
        if sid is not None:
            all_h1_sids.add(int(sid))
    most_recent_h1_sid = max(all_h1_sids) if all_h1_sids else 0

    rev_confirmed_by_sid = _get_reversal_confirmed_by_sid(structure_events)

    # --- H1 breakout labels ("bo") ---
    if state_cfg.get("labels", False) or True:  # Always show H1 labels on M15 chart
        bo_events = [
            ev for ev in structure_events
            if ev.type == "STATE_CHANGED" and ev.meta.get("to") == "breakout"
            and ev.idx in h1_df.index
        ]
        if bo_events:
            x_vals, y_vals, cd = [], [], []
            for ev in bo_events:
                m15_t = _h1_idx_to_m15_time(ev.idx)
                if m15_t is None:
                    continue
                sd = int(ev.meta.get("struct_direction", 1))
                sid = int(ev.meta.get("structure_id", 0))
                row = h1_df.loc[ev.idx]
                wo = (float(row[COL_H]) - float(row[COL_L])) * 0.15
                y = float(row[COL_H]) + wo * 2 if sd == 1 else float(row[COL_L]) - wo * 2
                x_vals.append(m15_t)
                y_vals.append(y)
                cd.append((ev.idx, "breakout", sid, sd))

            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode="text", text=["bo"] * len(x_vals),
                name="H1 bo", textposition="middle center", showlegend=False,
                hovertemplate=(
                    "TF=1H<br>idx=%{customdata[0]}<br>state=%{customdata[1]}<br>"
                    "sid=%{customdata[2]}<br>struct_direction=%{customdata[3]}<extra></extra>"
                ),
                customdata=cd,
            ))

        # --- H1 state labels (pb/pr/rv) ---
        label_map = {"pullback": "pb", "pullback_range": "pr", "reversal": "rv"}
        state_events = [
            ev for ev in structure_events
            if ev.type == "STATE_CHANGED" and ev.meta.get("to") in label_map
            and ev.idx in h1_df.index
        ]
        if state_events:
            x_vals, y_vals, text_vals, cd = [], [], [], []
            for ev in state_events:
                m15_t = _h1_idx_to_m15_time(ev.idx)
                if m15_t is None:
                    continue
                sd = int(ev.meta.get("struct_direction", 1))
                sid = int(ev.meta.get("structure_id", 0))
                row = h1_df.loc[ev.idx]
                wo = (float(row[COL_H]) - float(row[COL_L])) * 0.15
                y = float(row[COL_L]) - wo * 2 if sd == 1 else float(row[COL_H]) + wo * 2
                x_vals.append(m15_t)
                y_vals.append(y)
                text_vals.append(label_map[ev.meta["to"]])
                cd.append((ev.idx, ev.meta["to"], sid, sd))

            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode="text", text=text_vals,
                name="H1 state labels", textposition="middle center", showlegend=False,
                hovertemplate=(
                    "TF=1H<br>idx=%{customdata[0]}<br>state=%{customdata[1]}<br>"
                    "sid=%{customdata[2]}<br>struct_direction=%{customdata[3]}<extra></extra>"
                ),
                customdata=cd,
            ))

    # --- H1 structure swing lines (dashed) ---
    if struct_cfg.get("levels", False):
        cts_events = [ev for ev in structure_events if ev.type == "CTS_CONFIRMED"]
        bos_events = [ev for ev in structure_events if ev.type == "BOS_CONFIRMED"]

        points_by_sid = defaultdict(list)
        for ev in cts_events:
            p_idx = ef.cts_anchor_idx(ev)
            m15_t = _h1_idx_to_m15_time(p_idx)
            if m15_t is None:
                continue
            price = float(ev.price) if ev.price is not None else 0.0
            if price == 0.0 and p_idx in h1_df.index and "cts_price" in h1_df.columns:
                price = float(h1_df.loc[p_idx, "cts_price"])
            sid = int(ev.meta.get("structure_id", 0))
            cycle = int(ev.meta.get("cycle_id", 0))
            sd = int(ev.meta.get("struct_direction", 0))
            points_by_sid[sid].append((p_idx, m15_t, price, "CTS", sid, cycle, sd))

        for ev in bos_events:
            b_idx = ef.bos_anchor_idx(ev)   # the BOS dot sits at its anchor
            m15_t = _h1_idx_to_m15_time(b_idx)
            if m15_t is None:
                continue
            price = float(ev.price) if ev.price is not None else 0.0
            sid = int(ev.meta.get("structure_id", 0))
            cycle = int(ev.meta.get("cycle_id", 0))
            sd = int(ev.meta.get("struct_direction", 0))
            points_by_sid[sid].append((b_idx, m15_t, price, "BOS", sid, cycle, sd))

        # Unconfirmed CTS + PB for H1
        cts_unconf = [ev for ev in structure_events if ev.type in ("CTS_ESTABLISHED", "CTS_UPDATED")]
        pb_state = [ev for ev in structure_events if ev.type == "STATE_CHANGED" and ev.meta.get("to") == "pullback"]
        pb_to_bos_lines = []

        for sid in sorted(all_h1_sids):
            sid_pts = sorted(points_by_sid.get(sid, []), key=lambda x: x[0])
            if not sid_pts:
                continue
            last_pt = sid_pts[-1]
            last_kind = last_pt[3]
            last_idx = last_pt[0]
            sd_for_sid = last_pt[6]

            if last_kind == "BOS":
                # The dot sits at the CTS ANCHOR — a location.
                cts_after = [e for e in cts_unconf
                             if int(e.meta.get("structure_id", -1)) == sid and ef.cts_anchor_idx(e) > last_idx]
                if cts_after:
                    latest = max(cts_after, key=ef.cts_anchor_idx)
                    latest_idx = ef.cts_anchor_idx(latest)
                    m15_t = _h1_idx_to_m15_time(latest_idx)
                    if m15_t is not None:
                        price = float(latest.price) if latest.price is not None else 0.0
                        cycle = int(latest.meta.get("cycle_id", 0))
                        points_by_sid[sid].append((latest_idx, m15_t, price, "CTS", sid, cycle, sd_for_sid))

            elif last_kind == "CTS" and sid != most_recent_h1_sid:
                next_sid = sid + 1
                next_bos = sorted(
                    [e for e in bos_events if int(e.meta.get("structure_id", -1)) == next_sid],
                    key=ef.event_moment,
                )
                # The PB search's upper bound is a TIME: the next sid's first BOS MOMENT
                # (Plan E E3g-3, PLAN_E §7.1 T4).
                next_bos_idx = ef.event_moment(next_bos[0]) if next_bos else None
                pb_after = [e for e in pb_state
                            if int(e.meta.get("structure_id", -1)) == sid
                            and int(e.idx) > last_idx   # a LOCATION lower bound: PBs after the last point's extreme
                            and (next_bos_idx is None or int(e.idx) < next_bos_idx)]
                if pb_after:
                    latest_pb = max(pb_after, key=lambda e: int(e.idx))
                    m15_t = _h1_idx_to_m15_time(latest_pb.idx)
                    if m15_t is not None:
                        if sd_for_sid == 1:
                            pb_price = float(h1_df.loc[latest_pb.idx, COL_L])
                        else:
                            pb_price = float(h1_df.loc[latest_pb.idx, COL_H])
                        points_by_sid[sid].append((latest_pb.idx, m15_t, pb_price, "PB", sid, 0, sd_for_sid))
                        if next_bos:
                            fb = next_bos[0]
                            fb_idx = ef.bos_anchor_idx(fb)   # the line's BOS end (location)
                            fb_t = _h1_idx_to_m15_time(fb_idx)
                            fb_price = float(fb.price) if fb.price is not None else 0.0
                            if fb_t is not None:
                                pb_to_bos_lines.append((sid, m15_t, pb_price, fb_t, fb_price,
                                                        int(latest_pb.idx), fb_idx))

        # Lifecycle filter (chart review 2026-09-21): on the SUB charts the H1
        # overlay draws only the waves that were live at some point. A wave —
        # a segment between two consecutive H1 points, or the most recent sid's
        # extension to the last candle — is drawn iff its candle span intersects
        # its sid's window `[overlay start, reversal idx]` — the start is a
        # LOCATION (`_h1_overlay_window_start_by_sid`: the first structural
        # anchor, or the reversal handoff; user decision 2026-09-25), the end
        # `compute_reversal_idx_by_sid`; waves wholly outside it (the retroactive (1,0)/(1,1)
        # waves of the post-reversal sid, which precede its 902 start on the
        # reference window) are NOT drawn — they only crowd the sub structures
        # they overlap. The H1 chart itself is unchanged (every sid, prior sids
        # dimmed). A dot is drawn iff a drawn wave or a drawn PB→BOS line
        # touches it.
        rev_h1 = compute_reversal_idx_by_sid(structure_events)
        overlay_start_h1 = _h1_overlay_window_start_by_sid(structure_events, rev_h1)
        drawn_h1_pts: set = set()   # {(sid, h1 idx)} endpoints of drawn waves
        for sid in sorted(points_by_sid.keys()):
            seq = sorted(points_by_sid[sid], key=lambda x: x[0])
            if sid == most_recent_h1_sid:
                last_h1_idx = int(h1_df.index[-1])
                end_t = _h1_idx_to_m15_time(last_h1_idx)
                if end_t is not None and seq and last_h1_idx > seq[-1][0]:
                    seq.append((last_h1_idx, end_t, float(h1_df[COL_C].iloc[-1]), "EXT", sid, -1, seq[-1][6]))
            # The window start is a LOCATION (the first anchor / the handoff).
            w_start = overlay_start_h1.get(sid, seq[0][0] if seq else None)
            w_end = rev_h1.get(sid)
            for is_live_run, i0, i1 in _split_polyline_by_wave([p[0] for p in seq], w_start, w_end):
                if not is_live_run:
                    continue          # never live — not drawn on the sub charts
                run = seq[i0:i1 + 1]
                drawn_h1_pts.update((p[4], p[0]) for p in run if p[3] != "EXT")
                line_style = _style("structure.h1_overlay.swing_line").copy()
                fig.add_trace(go.Scatter(
                    x=[p[1] for p in run], y=[p[2] for p in run], mode="lines",
                    name=f"H1 swing sid={sid}", hoverinfo="skip",
                    line_shape="linear", showlegend=False, **line_style,
                ))

        # Cross-structure PB→BOS lines belong to the PRIOR sid — same wave rule
        # on its window (drawn on the reference window: sid 0's 683→689).
        for _pb_sid, pb_t, pb_p, bos_t, bos_p, pb_idx, bos_idx in pb_to_bos_lines:
            if not _wave_touches_window(pb_idx, bos_idx, overlay_start_h1.get(_pb_sid), rev_h1.get(_pb_sid)):
                continue
            drawn_h1_pts.add((_pb_sid, pb_idx))
            drawn_h1_pts.add((_pb_sid + 1, bos_idx))
            line_style = _style("structure.h1_overlay.swing_line").copy()
            fig.add_trace(go.Scatter(
                x=[pb_t, bos_t], y=[pb_p, bos_p], mode="lines",
                name=f"H1 PB→BOS sid={_pb_sid}", hoverinfo="skip",
                line_shape="linear", showlegend=False, **line_style,
            ))

        # H1 CTS/BOS dots — only the points a drawn wave / PB→BOS line touches.
        all_pts = []
        for sid_pts in points_by_sid.values():
            all_pts.extend(p for p in sid_pts if (p[4], p[0]) in drawn_h1_pts)
        for kind_filter, style_key in [("CTS", "structure.cts"), ("BOS", "structure.bos"), ("PB", "structure.bos")]:
            pts = [p for p in all_pts if p[3] == kind_filter or (kind_filter == "CTS" and p[3].startswith("CTS"))]
            if not pts:
                continue
            style = _style(style_key).copy()
            cd = [[p[0], p[3], p[2], p[4], p[5], p[6]] for p in pts]
            fig.add_trace(go.Scatter(
                x=[p[1] for p in pts], y=[p[2] for p in pts],
                mode="markers", name=f"H1 {kind_filter}",
                showlegend=False, customdata=cd,
                hovertemplate=(
                    "TF=1H<br>idx=%{customdata[0]}<br>kind=%{customdata[1]}<br>"
                    "price=%{customdata[2]:.5f}<br>sid=%{customdata[3]}<br>"
                    "cycle_id=%{customdata[4]}<br>struct_direction=%{customdata[5]}<extra></extra>"
                ),
                **style,
            ))

    # --- H1 KL zones (color fill) ---
    h1_kl_zones = h1_df.attrs.get("kl_zones", [])
    if zone_cfg.get("KL", False) and h1_kl_zones:
        by_struct = {}
        for z in h1_kl_zones:
            sid = int(z.meta.get("structure_id", 0))
            by_struct.setdefault(sid, []).append(z)

        all_kl_sids = sorted(by_struct.keys(), reverse=True)
        num_structures = int(zone_cfg.get("num_structures", 99))
        selected_kl_sids = set(all_kl_sids[:num_structures])
        most_recent_kl_sid = all_kl_sids[0] if all_kl_sids else 0

        # Per-TF tier for H1 zones on M15 chart = main_tf (0.2).
        h1_overlay_tier = select_subordinate_tf_tier("H1", primary_sub_tf="M15")

        for z in h1_kl_zones:
            zone_sid = int(z.meta.get("structure_id", 0))
            if zone_sid not in selected_kl_sids:
                continue
            if is_collapsed_cycle_zone(z):
                continue          # collapsed (retroactive) cycle — never active; not drawn

            side = str(z.side)
            # Read unified base values from zone.kl.* (Item 5: legacy
            # zone.h1_overlay.kl.* bases supersede per-TF tier multiplication).
            stz = STYLE.get(f"zone.kl.{side}", {})

            rgb = str(stz.get("rgb", "0,180,0" if side == "buy" else "220,0,0"))
            base_fill_op = float(stz.get("fill_opacity_active", 0.4))
            base_line_op = float(stz.get("confirm_opacity_active", 0.9))
            fill_op = base_fill_op * h1_overlay_tier
            confirm_op = base_line_op * h1_overlay_tier
            confirm_w = int(stz.get("confirm_line_width", 2))

            fillcolor = _rgba_from_rgb(rgb, fill_op)
            confirm_color = _rgba_from_rgb(rgb, confirm_op)

            x0 = _h1_to_m15_time(pd.to_datetime(z.start_time, utc=True))
            x1 = _h1_to_m15_time(pd.to_datetime(z.end_time, utc=True)) if z.end_time else m15_t_last

            conf_idx = int(z.meta.get("confirmed_idx", -1))
            conf_time = _h1_idx_to_m15_time(conf_idx) if conf_idx >= 0 else None

            steps = list((z.meta or {}).get("bounds_steps", []))
            fallback_top = float(max(z.top, z.bottom))
            fallback_bot = float(min(z.top, z.bottom))
            if not steps:
                steps = [{"start_idx": int(z.meta.get("base_idx", 0)),
                          "top": fallback_top,
                          "bottom": fallback_bot,
                          "event": "FALLBACK"}]
            steps = sorted(steps, key=lambda s: int(s.get("start_idx", -1)))

            # render_end_idx in H1 idx space.
            if z.end_time is None:
                render_end_idx = int(h1_df.index[-1])
            else:
                _em = h1_times <= pd.to_datetime(z.end_time, utc=True)
                render_end_idx = int(h1_df.index[_em][-1]) if _em.any() else int(h1_df.index[0])

            active_stretches = compute_kl_active_stretches(z, render_end_idx)

            # Per-step iteration: fills (only active stretch) + hover lines.
            for k, s in enumerate(steps):
                seg_start_h1 = int(s.get("start_idx", -1))
                seg_x0 = _h1_idx_to_m15_time(seg_start_h1)
                if seg_x0 is None:
                    continue
                if seg_x0 < x0:
                    seg_x0 = x0

                if k + 1 < len(steps):
                    next_h1 = int(steps[k + 1].get("start_idx", -1))
                    nxt = _h1_idx_to_m15_time(next_h1) or x1
                    seg_x1 = nxt - pd.Timedelta(microseconds=1)
                    step_end_h1 = next_h1 - 1
                else:
                    seg_x1 = x1
                    step_end_h1 = render_end_idx

                seg_top = float(s.get("top", z.top))
                seg_bot = float(s.get("bottom", z.bottom))
                sy0 = min(seg_bot, seg_top)
                sy1 = max(seg_bot, seg_top)

                for stretch_start, stretch_end in active_stretches:
                    isect_start = max(seg_start_h1, stretch_start)
                    isect_end = min(step_end_h1, stretch_end)
                    if isect_start > isect_end:
                        continue
                    fill_x0 = _h1_idx_to_m15_time(isect_start) or seg_x0
                    if fill_x0 < seg_x0:
                        fill_x0 = seg_x0
                    if isect_end >= step_end_h1:
                        fill_x1 = seg_x1
                    else:
                        end_time = _h1_idx_to_m15_time(isect_end)
                        fill_x1 = end_time if end_time is not None else seg_x1
                    if fill_x1 <= fill_x0:
                        continue
                    fig.add_shape(
                        type="rect", xref="x", yref="y",
                        x0=fill_x0, x1=fill_x1, y0=sy0, y1=sy1,
                        fillcolor=fillcolor,
                        line=dict(width=0),
                        layer="below",
                    )

                # Hover lines per step (preserved)
                seg_times = dfx[COL_TIME][(dfx[COL_TIME] >= seg_x0) & (dfx[COL_TIME] <= seg_x1)]
                if len(seg_times) == 0:
                    seg_times = pd.Series([seg_x0, seg_x1])

                hover_cd = [[
                    side, int(z.meta.get("structure_id", -1)),
                    int(z.meta.get("struct_direction", 0)),
                    str(z.meta.get("base_pattern", "")),
                    int(z.meta.get("base_idx", -1)),
                    conf_idx, int(z.meta.get("cycle_id", 0)),
                    sy1, sy0,
                ]] * len(seg_times)

                h1_hover_line = {"width": 6, "color": "rgba(0,0,0,0)"}
                for yval in (sy1, sy0):
                    fig.add_trace(go.Scatter(
                        x=seg_times, y=[yval] * len(seg_times),
                        mode="lines", name="H1 KL zone (overlay)", showlegend=False,
                        line=h1_hover_line, line_shape="hv",
                        hovertemplate=(
                            "TF=1H<br>"
                            "KL Zone<br>"
                            "side=%{customdata[0]}<br>"
                            "structure_id=%{customdata[1]}<br>"
                            "struct_direction=%{customdata[2]}<br>"
                            "base_pattern=%{customdata[3]}<br>"
                            "base_idx=%{customdata[4]}<br>"
                            "confirmed_idx=%{customdata[5]}<br>"
                            "cycle_id=%{customdata[6]}<br>"
                            "top=%{customdata[7]:.5f}<br>"
                            "bottom=%{customdata[8]:.5f}"
                            "<extra></extra>"
                        ),
                        customdata=hover_cd,
                    ))

            # Stepped polygon outline (main-TF overlay: confirm color, width 2).
            outline_xs, outline_ys = build_stepped_outline_xy(
                steps, _h1_idx_to_m15_time, x0, x1,
                fallback_top=fallback_top, fallback_bottom=fallback_bot,
            )
            fig.add_trace(go.Scatter(
                x=outline_xs, y=outline_ys, mode="lines",
                line=dict(color=confirm_color, width=confirm_w),
                fill=None, hoverinfo="skip", showlegend=False,
                name=f"H1 KL outline (overlay) sid={zone_sid}",
            ))

            # Single confirm line at confirmed_idx — position in step containing it.
            if conf_time is not None and x0 <= conf_time <= x1:
                for k, s in enumerate(steps):
                    seg_x0 = _h1_idx_to_m15_time(int(s.get("start_idx", -1)))
                    if seg_x0 is None:
                        continue
                    if seg_x0 < x0:
                        seg_x0 = x0
                    if k + 1 < len(steps):
                        nxt = _h1_idx_to_m15_time(int(steps[k + 1].get("start_idx", -1))) or x1
                        seg_x1 = nxt - pd.Timedelta(microseconds=1)
                    else:
                        seg_x1 = x1
                    if seg_x0 <= conf_time <= seg_x1:
                        seg_top = float(s.get("top", z.top))
                        seg_bot = float(s.get("bottom", z.bottom))
                        fig.add_shape(
                            type="line", xref="x", yref="y",
                            x0=conf_time, x1=conf_time,
                            y0=min(seg_bot, seg_top), y1=max(seg_bot, seg_top),
                            line=dict(color=confirm_color, width=confirm_w),
                            layer="below",
                        )
                        break

    # --- H1 Wave candle verticals (dashed; per-candle lifecycle gating) ---
    h1_wave_candles = h1_df.attrs.get("wave_candles", [])
    if zone_cfg.get("wave_candles", True) and h1_wave_candles:
        from engine_v2.zones.structure_lifecycle import compute_cycle_lifecycle
        from engine_v2.zones.wave_candles import (
            _role_for_wave_candle,
            compute_wave_candle_visibility,
        )

        # Per-cycle lifecycle from H1 main events (no floor/cap).
        h1_events = h1_df.attrs.get("structure_events", [])
        rev_h1 = compute_reversal_idx_by_sid(h1_events)
        struct_floor_h1 = compute_struct_start_by_sid(h1_events, rev_h1)
        cycle_life_h1 = compute_cycle_lifecycle(h1_events, rev_h1)
        cts_conf_h1: dict = {}
        for ev in h1_events:
            if getattr(ev, "type", None) != "CTS_CONFIRMED":
                continue
            m = ev.meta or {}
            s = m.get("structure_id"); c = m.get("cycle_id")
            if s is None or c is None:
                continue
            s, c = int(s), int(c)
            idx = int(ev.idx)
            sfloor = struct_floor_h1.get(s)
            if sfloor is not None:
                idx = max(idx, int(sfloor))
            cts_conf_h1[(s, c)] = idx
        h1_wc_visibility = compute_wave_candle_visibility(cycle_life_h1, cts_conf_h1)

        wc_y_min = float(dfx[COL_L].min())
        wc_y_max = float(dfx[COL_H].max())

        for wc in h1_wave_candles:
            for position, idx in (("last", wc.last_wave_candle_idx),
                                  ("first", wc.first_wave_candle_idx)):
                if idx is None or idx not in h1_df.index:
                    continue
                role_info = _role_for_wave_candle(str(wc.source_kind), position)
                if role_info is None:
                    continue
                role, cycle_offset = role_info
                lookup_cycle = wc.cycle_id + cycle_offset
                if lookup_cycle < 0:
                    # BOS_0.last (pre-structure pullback). Option C
                    # (2026-05-28): render iff this sid's cyc=0 is
                    # non-collapsed.
                    life0 = cycle_life_h1.get((wc.structure_id, 0))
                    if life0 is None:
                        continue
                    cstart0, cend0, _r0 = life0
                    if cstart0 is None or (cend0 is not None and cstart0 >= cend0):
                        continue
                    # cyc=0 non-collapsed → render.
                else:
                    viz = h1_wc_visibility.get((wc.structure_id, lookup_cycle, role))
                    if viz is None or not viz[0]:
                        continue
                candle_dir = int(h1_df.loc[idx, "direction"])
                if candle_dir == 0:
                    continue

                style_key = "wave_candle.h1_overlay.bullish" if candle_dir == 1 else "wave_candle.h1_overlay.bearish"
                wc_style = _style(style_key)
                line_info = wc_style.get("line", {})
                color_rgb = line_info.get("color_rgb", "128,128,128")
                base_opacity = float(wc_style.get("opacity", 0.8))
                line_width = int(line_info.get("width", 1))
                dash = line_info.get("dash")

                # Uniform opacity (no tier multiplier).
                final_color = f"rgba({color_rgb}, {base_opacity})"
                wc_time = _h1_idx_to_m15_time(idx)
                if wc_time is None:
                    continue

                line_props = dict(color=final_color, width=line_width)
                if dash:
                    line_props["dash"] = dash
                fig.add_shape(type="line", xref="x", yref="paper",
                              x0=wc_time, x1=wc_time, y0=0, y1=1,
                              line=line_props,
                              layer="below")

                # Wave-candle hover (no WVMI momentum / Weighted vol).
                vol = float(h1_df.loc[idx, "volume"])
                hover_lines = [
                    "TF=1H",
                    "<b>Wave Candle</b>",
                    f"idx={idx}",
                    # TODO: label hardcodes "BOS zone:" but CTS wave candles render here too.
                    f"BOS zone: sid={wc.structure_id} cycle={wc.cycle_id}",
                    f"Volume: {vol:.0f}",
                ]
                _n_pts = 12
                _y_pts = [wc_y_min + i * (wc_y_max - wc_y_min) / (_n_pts - 1) for i in range(_n_pts)]
                fig.add_trace(go.Scatter(
                    x=[wc_time] * _n_pts, y=_y_pts,
                    mode="lines", showlegend=False,
                    line=dict(width=8, color="rgba(0,0,0,0)"),
                    hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
                ))

    # --- H1 POI zones (color fill) ---
    h1_poi_zones = h1_df.attrs.get("poi_zones", [])
    if zone_cfg.get("POI", False) and h1_poi_zones:
        all_poi_sids = set(int(z.meta.get("structure_id", 0)) for z in h1_poi_zones)
        most_recent_poi_sid = max(all_poi_sids) if all_poi_sids else 0

        # Per-TF tier for H1 POI on M15 chart = main_tf (0.2).
        h1_overlay_tier_poi = select_subordinate_tf_tier("H1", primary_sub_tf="M15")

        _collapsed_h1 = collapsed_cycles(h1_kl_zones)
        for poi in h1_poi_zones:
            if poi.meta.get("status") == "disappeared":
                continue
            if is_poi_of_collapsed_cycle(poi, _collapsed_h1):
                continue

            side = str(poi.side)
            # Read unified base values from zone.poi.* (Item 5).
            stz = STYLE.get(f"zone.poi.{side}", {})
            zone_status = poi.meta.get("status", "active")
            zone_sid = int(poi.meta.get("structure_id", 0))

            rgb = str(stz.get("rgb", "255, 215, 0"))
            base_fill_op = float(stz.get("fill_opacity_active", 0.9))
            fill_op = base_fill_op * h1_overlay_tier_poi
            confirm_rgb = str(stz.get("confirm_line_rgb", "101, 67, 33"))
            confirm_op = float(stz.get("confirm_opacity_active", 0.9)) * h1_overlay_tier_poi
            confirm_w = int(stz.get("confirm_line_width", 2))

            fillcolor = _rgba_from_rgb(rgb, fill_op)
            confirm_color = _rgba_from_rgb(confirm_rgb, confirm_op)

            x0 = _h1_to_m15_time(pd.to_datetime(poi.start_time, utc=True))
            x1 = _h1_to_m15_time(pd.to_datetime(poi.end_time, utc=True)) if poi.end_time else m15_t_last

            y0 = float(min(poi.top, poi.bottom))
            y1 = float(max(poi.top, poi.bottom))

            _conf_raw = poi.meta.get("confirmed_idx")
            conf_idx = int(_conf_raw) if _conf_raw is not None else None

            # render_end_idx in H1 idx space.
            if poi.end_time is None:
                render_end_idx_h1 = int(h1_df.index[-1])
            else:
                _em = h1_times <= pd.to_datetime(poi.end_time, utc=True)
                render_end_idx_h1 = int(h1_df.index[_em][-1]) if _em.any() else int(h1_df.index[0])

            poi_stretches = compute_poi_active_stretches(poi, render_end_idx_h1)

            # Active-stretch fills.
            for stretch_start, stretch_end in poi_stretches:
                sx0 = _h1_idx_to_m15_time(stretch_start)
                sx1 = _h1_idx_to_m15_time(stretch_end)
                if sx0 is None or sx1 is None:
                    continue
                if sx0 < x0:
                    sx0 = x0
                if sx1 > x1:
                    sx1 = x1
                if sx1 <= sx0:
                    continue
                fig.add_shape(
                    type="rect", xref="x", yref="y",
                    x0=sx0, x1=sx1, y0=y0, y1=y1,
                    fillcolor=fillcolor,
                    line=dict(width=0),
                    layer="below",
                )

            # Outline rect (no fill, main-TF overlay: brown border width 2).
            fig.add_shape(
                type="rect", xref="x", yref="y",
                x0=x0, x1=x1, y0=y0, y1=y1,
                fillcolor="rgba(0,0,0,0)",
                line=dict(color=confirm_color, width=confirm_w),
                layer="below",
            )

            # N vertical confirm lines from activation_history (fallback: confirmed_idx).
            activation_history = poi.meta.get("activation_history", []) or []
            confirm_idxs_h1 = [
                int(ev["idx"]) for ev in activation_history
                if ev.get("active") and int(ev["idx"]) <= render_end_idx_h1
            ]
            if not confirm_idxs_h1 and conf_idx is not None:
                confirm_idxs_h1 = [conf_idx]
            for c_idx in confirm_idxs_h1:
                c_time = _h1_idx_to_m15_time(c_idx)
                if c_time is None or not (x0 <= c_time <= x1):
                    continue
                fig.add_shape(
                    type="line", xref="x", yref="y",
                    x0=c_time, x1=c_time, y0=y0, y1=y1,
                    line=dict(color=confirm_color, width=confirm_w),
                    layer="below",
                )

            seg_times = dfx[COL_TIME][(dfx[COL_TIME] >= x0) & (dfx[COL_TIME] <= x1)]
            if len(seg_times) == 0:
                seg_times = pd.Series([x0, x1])

            versions = poi.meta.get("versions", [])
            versions_str = ", ".join(versions) if versions else "none"
            poi_cd = [[
                side, poi.meta.get("structure_id", -1), poi.meta.get("struct_direction", 0),
                poi.ic_idx, conf_idx, poi.meta.get("cycle_id", 0),
                versions_str, y1, y0, zone_status,
            ]] * len(seg_times)

            h1_hover_line = {"width": 6, "color": "rgba(0,0,0,0)"}
            for yval in (y1, y0):
                fig.add_trace(go.Scatter(
                    x=seg_times, y=[yval] * len(seg_times),
                    mode="lines", name="H1 POI zone (overlay)", showlegend=False,
                    line=h1_hover_line, line_shape="hv",
                    hovertemplate=(
                        "TF=1H<br>"
                        "<b>POI Zone</b><br>"
                        "side=%{customdata[0]}<br>"
                        "structure_id=%{customdata[1]}<br>"
                        "struct_direction=%{customdata[2]}<br>"
                        "ic_idx=%{customdata[3]}<br>"
                        "confirmed_idx=%{customdata[4]}<br>"
                        "cycle_id=%{customdata[5]}<br>"
                        "versions=%{customdata[6]}<br>"
                        "top=%{customdata[7]:.5f}<br>"
                        "bottom=%{customdata[8]:.5f}<br>"
                        "status=%{customdata[9]}"
                        "<extra></extra>"
                    ),
                    customdata=poi_cd,
                ))

    # --- H1 Prev BOS lines (dashed) ---
    h1_prev_bos = h1_df.attrs.get("prev_bos_lines", [])
    for line_info in h1_prev_bos:
        start_idx = line_info["start_idx"]
        end_idx = line_info["end_idx"]
        price = line_info["price"]
        t0 = _h1_idx_to_m15_time(start_idx)
        t1 = _h1_idx_to_m15_time(end_idx)
        if t0 is not None and t1 is not None:
            fig.add_trace(go.Scatter(
                x=[t0, t1], y=[price, price], mode="lines",
                line=_style("prev_bos_line.h1_overlay").get("line", {"width": 2, "color": "black", "dash": "dash"}),
                name=f"H1 Prev BOS (sid={line_info.get('prev_structure_id', '?')})",
                showlegend=False,
                hovertemplate=(
                    f"TF=1H<br>"
                    f"Prev BOS Line<br>"
                    f"Price: {price:.5f}<br>"
                    f"From idx: {start_idx}<br>"
                    f"To idx: {end_idx}<br>"
                    f"<extra></extra>"
                ),
            ))


# ---------------------------------------------------------------------------
# Phase C+: H1 zone-proximity triggers, anchored at the matching M15 candle
# ---------------------------------------------------------------------------

def _render_proximity_triggers_overlay(fig, dfx, h1_df, wick_offset):
    """Render H1 zone_proximity_triggers on the M15 chart.

    Each H1 trigger fires at an H1 candle whose wick reached within
    threshold of a zone inner. The trigger gets anchored on the M15
    sub-candle whose extreme matches that H1 wick:
      - approach_from_above (wick is the candle low) -> M15 with min low
      - approach_from_below (wick is the candle high) -> M15 with max high

    Marker style is shared with the H1 chart via `zone_proximity.trigger`.
    The hover panel reports both M15 and H1 idx so the trigger is
    diagnosable from the M15 view alone.
    """
    zone_proximity_triggers = h1_df.attrs.get("zone_proximity_triggers", {})
    if not zone_proximity_triggers:
        return

    zpt_style = _style("zone_proximity.trigger")
    zpt_marker = zpt_style.get("marker", {"size": 7, "symbol": "x", "color": "black"})
    offset_mult = float(zpt_style.get("offset_mult", 2.5))

    h1_times = pd.to_datetime(h1_df[COL_TIME], utc=True)

    # struct_direction lookup by (sid, cycle_id) from H1 CTS_CONFIRMED events.
    structure_events = h1_df.attrs.get("structure_events", [])
    sd_by_cycle: dict = {}
    for ev in structure_events:
        if getattr(ev, "type", None) != "CTS_CONFIRMED":
            continue
        sid = ev.meta.get("structure_id")
        cyc = ev.meta.get("cycle_id")
        sd = ev.meta.get("struct_direction")
        if sid is not None and cyc is not None and sd is not None:
            sd_by_cycle[(int(sid), int(cyc))] = int(sd)

    # H1 KL zone confirmed_idx lookup for hover.
    kl_conf_idx_by_key: dict = {}
    for z in h1_df.attrs.get("kl_zones", []):
        sid = z.meta.get("structure_id")
        cyc = z.meta.get("cycle_id")
        sk = z.source_kind
        cidx = z.meta.get("confirmed_idx")
        if sid is not None and cyc is not None and sk is not None and cidx is not None:
            kl_conf_idx_by_key[(int(sid), int(cyc), str(sk))] = int(cidx)

    poi_zones_by_cycle: dict = {}
    for pz in h1_df.attrs.get("poi_zones", []):
        sid = pz.meta.get("structure_id")
        cyc = pz.meta.get("cycle_id")
        if sid is None or cyc is None:
            continue
        poi_zones_by_cycle.setdefault((int(sid), int(cyc)), []).append(pz)

    def _poi_confirmed_idx(sid: int, cyc: int, inner: float, sd: int, at_idx: int) -> int:
        cands = poi_zones_by_cycle.get((sid, cyc), [])
        best = None
        best_diff = float("inf")
        for pz in cands:
            pz_inner = float(pz.top) if sd == 1 else float(pz.bottom)
            diff = abs(pz_inner - inner)
            if diff < best_diff:
                best_diff = diff
                best = pz
        if best is None:
            return -1
        cidx = poi_confirmed_idx_as_of(best, at_idx)
        return int(cidx) if cidx is not None else -1

    # M15 times (already datetime in dfx; build once for binary lookup).
    m15_times = pd.to_datetime(dfx[COL_TIME], utc=True)

    def _find_m15_by_extreme(h1_time: pd.Timestamp, approach_from_above: bool):
        """Return the M15 dfx idx whose wick extreme matches the H1 trigger.

        approach_from_above=True -> trigger wick is the H1 candle low,
        so pick the M15 candle with the lowest low (tie -> last).
        approach_from_above=False -> trigger wick is the H1 candle high,
        so pick the M15 candle with the highest high (tie -> last).
        Returns None if no M15 candles fall inside [h1_time, h1_time+1h).
        """
        h1_end = h1_time + timedelta(hours=1)
        mask = (m15_times >= h1_time) & (m15_times < h1_end)
        candidates = dfx[mask]
        if candidates.empty:
            return None
        if approach_from_above:
            best_val = candidates[COL_L].min()
            matches = candidates[candidates[COL_L] == best_val]
        else:
            best_val = candidates[COL_H].max()
            matches = candidates[candidates[COL_H] == best_val]
        return int(matches.index[-1])

    x_vals, y_vals, customdata = [], [], []
    skipped_no_m15 = 0
    for (sid_c, cyc_c), trig_list in zone_proximity_triggers.items():
        sd = sd_by_cycle.get((int(sid_c), int(cyc_c)), 0)
        for trig in trig_list:
            h1_idx = int(trig.idx)
            if h1_idx not in h1_df.index:
                continue
            h1_time = pd.to_datetime(h1_df.loc[h1_idx, COL_TIME], utc=True)
            h1_row = h1_df.loc[h1_idx]

            approach_from_above = (
                (sd == 1 and trig.direction == "sd")
                or (sd == -1 and trig.direction == "opp_sd")
            )

            m15_idx = _find_m15_by_extreme(h1_time, approach_from_above)
            if m15_idx is None or m15_idx not in dfx.index:
                skipped_no_m15 += 1
                continue

            m15_row = dfx.loc[m15_idx]
            wo = float(wick_offset.loc[m15_idx]) if m15_idx in wick_offset.index else 0.0
            o, c = float(m15_row[COL_O]), float(m15_row[COL_C])
            if c < o:
                y = float(m15_row[COL_H]) + wo * offset_mult
            else:
                y = float(m15_row[COL_L]) - wo * offset_mult

            type_label = f"{trig.direction}:{trig.zone_kind}"

            # Distance from H1 trigger wick to zone inner (parent-TF metric).
            if approach_from_above:
                gap = float(h1_row[COL_L]) - float(trig.trigger_inner)
            else:
                gap = float(trig.trigger_inner) - float(h1_row[COL_H])
            dist_pips = round(gap / float(trig.pip_size), 1)

            if trig.zone_kind == "POI":
                z_conf = _poi_confirmed_idx(
                    int(trig.structure_id), int(trig.cycle_id),
                    float(trig.trigger_inner), int(sd) if sd else 1, h1_idx,
                )
            else:
                z_conf = kl_conf_idx_by_key.get(
                    (int(trig.structure_id), int(trig.cycle_id), str(trig.zone_kind)),
                    -1,
                )

            x_vals.append(m15_row[COL_TIME])
            y_vals.append(y)
            customdata.append((
                int(m15_idx),
                h1_idx,
                int(trig.structure_id),
                int(sd),
                int(trig.cycle_id),
                type_label,
                dist_pips,
                str(trig.zone_kind),
                int(z_conf),
            ))

    if x_vals:
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            name="zone_proximity:trigger",
            marker=zpt_marker,
            customdata=customdata,
            hoverlabel=dict(bgcolor="black", font_color="white"),
            hovertemplate=(
                "TF=1H>>15M<br>"
                "idx_15M=%{customdata[0]}<br>"
                "idx_1H=%{customdata[1]}<br>"
                "parent_sid=%{customdata[2]}<br>"
                "parent struct_direction=%{customdata[3]}<br>"
                "parent_cycle_id=%{customdata[4]}<br>"
                "type=%{customdata[5]}<br>"
                "distance_pips=%{customdata[6]}<br>"
                "zone_kind=%{customdata[7]}<br>"
                "zone_confirmed_idx=%{customdata[8]}"
                "<extra></extra>"
            ),
        ))
        print(f"[m15_chart][zone_proximity] rendered {len(x_vals)} trigger markers"
              + (f" (skipped {skipped_no_m15} with no M15 candle in window)"
                 if skipped_no_m15 else ""))


# ---------------------------------------------------------------------------
# JS injection for volume auto-rescale
# ---------------------------------------------------------------------------

def _inject_volume_autoscale_js(html_path: Path):
    js = """
<script>
(function initVolumeAutoscale() {
    var gd = document.querySelector('.plotly-graph-div');
    if (!gd || !gd._fullData) {
        setTimeout(initVolumeAutoscale, 200);
        return;
    }
    var volumeTraceIdx = -1;
    for (var i = 0; i < gd._fullData.length; i++) {
        if (gd._fullData[i].type === 'bar' && gd._fullData[i].yaxis === 'y2') {
            volumeTraceIdx = i;
            break;
        }
    }
    if (volumeTraceIdx === -1) return;
    var volumeX = gd._fullData[volumeTraceIdx].x;
    var volumeY = gd._fullData[volumeTraceIdx].y;
    function rescaleVolumeY() {
        var xRange = gd.layout.xaxis.range;
        if (!xRange) return;
        var x0 = new Date(xRange[0]).getTime();
        var x1 = new Date(xRange[1]).getTime();
        var maxVol = 0;
        for (var i = 0; i < volumeX.length; i++) {
            var t = new Date(volumeX[i]).getTime();
            if (t >= x0 && t <= x1) {
                if (volumeY[i] > maxVol) maxVol = volumeY[i];
            }
        }
        if (maxVol > 0) {
            Plotly.relayout(gd, {'yaxis2.range': [0, maxVol * 1.1]});
        }
    }
    gd.on('plotly_relayout', function(eventData) {
        if (eventData['xaxis.range[0]'] !== undefined ||
            eventData['xaxis.range'] !== undefined ||
            eventData['xaxis.autorange'] !== undefined) {
            setTimeout(rescaleVolumeY, 100);
        }
    });
    rescaleVolumeY();
})();
</script>
"""
    with open(html_path, "a") as f:
        f.write(js)
