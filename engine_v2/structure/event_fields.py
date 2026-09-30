"""Named reads of a structure event's candle indices (Plan E; ARCHITECTURE
"`ev.idx` convention"; GLOSSARY "Naming Standard").

A CTS / BOS event carries two candle indices with different roles:

- the ANCHOR — the price location adopted as the element's endpoint
  (`cts_anchor_idx`, `bos_anchor_idx`), and the pattern-realm anchor
  (`pattern_anchor_idx`, the breakout pattern's first candle);
- the MOMENT — the candle at which the event became knowable
  (`event_moment`).

Every production read of a `CTS_ESTABLISHED` / `BOS_CONFIRMED` index names its
role through this module, so the E4 flip (`ev.idx` := the moment; E4a
`CTS_ESTABLISHED`, E4b `BOS_CONFIRMED`) changes no reader. Direct indexing
only: every emitter writes these keys (event contract, LANDMINES "Event
Contract Rules" rule 3) — never `.get(key, ev.idx)`.

Call QUALIFIED — `from engine_v2.structure import event_fields as ef`;
`ef.cts_anchor_idx(ev)`. A direct import would collide with the many locals and
parameters of the same names (`cts_anchor_idx = cts_anchor_idx(ev)` raises
`UnboundLocalError`); `tests/test_event_fields.py` bans it.

Imports nothing from `market_structure` (it imports this module), so no cycle.
"""

from __future__ import annotations

from typing import Any, Tuple


# `CTS_UPDATED.meta["via"]` of the RAW path (`_maybe_update_cts_pre_confirm`): a
# new extreme seen on the processing candle, so `ev.idx` IS that candle. Every
# other `via` is a breakout-pattern name, whose `ev.idx` is the pattern's apply
# candle, the moment (Plan E E4c); its anchor is `meta["cts_anchor_idx"]`.
CTS_UPDATED_RAW_VIA = "replay_raw"

_CTS_ANCHOR_META_TYPES = ("CTS_ESTABLISHED", "CTS_CONFIRMED", "CTS_RECONFIRMED")
_PATTERN_ANCHOR_TYPES = ("CTS_ESTABLISHED", "REVERSAL_CANDIDATE", "REVERSAL_WATCH_START")


def cts_anchor_idx(ev: Any) -> int:
    """The CTS endpoint (a price location) the event refers to.

    - CTS_ESTABLISHED / CTS_CONFIRMED / CTS_RECONFIRMED: `meta["cts_anchor_idx"]`
      (a CONFIRMED / RECONFIRMED emitted with no current CTS carries None — a
      state the engine never reaches — and raises TypeError here, loudly).
    - CTS_UPDATED: `ev.idx` on the raw path (`via == CTS_UPDATED_RAW_VIA`, the
      processing candle — anchor and moment at once); `meta["cts_anchor_idx"]`
      on the pattern path (its `ev.idx` is the moment since Plan E E4c).

    Any other type raises.
    """
    if ev.type in _CTS_ANCHOR_META_TYPES:
        return int(ev.meta["cts_anchor_idx"])
    if ev.type == "CTS_UPDATED":
        if ev.meta["via"] == CTS_UPDATED_RAW_VIA:
            return int(ev.idx)
        return int(ev.meta["cts_anchor_idx"])
    raise ValueError(f"cts_anchor_idx: not a CTS event: {ev.type}")


def bos_anchor_idx(ev: Any) -> int:
    """The BOS endpoint (the swing extreme) of a BOS_CONFIRMED:
    `meta["bos_anchor_idx"]`. Any other type raises."""
    if ev.type == "BOS_CONFIRMED":
        return int(ev.meta["bos_anchor_idx"])
    raise ValueError(f"bos_anchor_idx: not a BOS_CONFIRMED: {ev.type}")


def pattern_anchor_idx(ev: Any) -> int:
    """The breakout pattern's FIRST candle (pattern realm):
    `meta["pattern_anchor_idx"]` on CTS_ESTABLISHED / REVERSAL_CANDIDATE /
    REVERSAL_WATCH_START. Any other type raises."""
    if ev.type in _PATTERN_ANCHOR_TYPES:
        return int(ev.meta["pattern_anchor_idx"])
    raise ValueError(f"pattern_anchor_idx: no pattern anchor on {ev.type}")


def event_moment(ev: Any) -> int:
    """The candle at which a CTS / BOS event became knowable (its MOMENT).

    - CTS_ESTABLISHED / BOS_CONFIRMED: `meta["confirmed_at"]` (== `ev.idx`
      since Plan E E4a / E4b).
    - CTS_CONFIRMED / CTS_RECONFIRMED: `ev.idx` (the confirmation candle).
    - CTS_UPDATED: `ev.idx` on the raw path (`via == CTS_UPDATED_RAW_VIA`, the
      processing candle); `meta["confirmed_at"]` on the pattern path (the
      pattern's apply candle, Plan E E3·0 — `== ev.idx` since E4c).
    - CTS_THRESHOLD_UPDATED / BOS_THRESHOLD_UPDATED: `ev.idx` (the processing
      candle — `_sync_thresholds_from_range` / `_bos_barrier_step` /
      the reversal-watch expiry (removed 2026-09-29) all stamp `i`; BOS added in Plan E E3g-2).
      (Until 2026-09-29 a reversal-watch expiry rewind replayed candles j < i and
      re-emitted threshold events stamped j although they were only known at i;
      MS no longer rewinds — F3b.)

    Any other type raises: a new consumer must define its event's moment.
    """
    if ev.type in ("CTS_ESTABLISHED", "BOS_CONFIRMED"):
        return int(ev.meta["confirmed_at"])
    if ev.type in ("CTS_CONFIRMED", "CTS_RECONFIRMED", "CTS_THRESHOLD_UPDATED", "BOS_THRESHOLD_UPDATED"):
        return int(ev.idx)
    if ev.type == "CTS_UPDATED":
        if ev.meta["via"] == CTS_UPDATED_RAW_VIA:
            return int(ev.idx)
        return int(ev.meta["confirmed_at"])
    raise ValueError(f"event_moment: no moment defined for {ev.type}")


def stamped_idx(ev: Any) -> int:
    """The index `ev.idx` held before the Plan E E4 flip, frozen against it: the
    anchor for CTS_ESTABLISHED / BOS_CONFIRMED / CTS_UPDATED (whose `ev.idx` is
    the moment since E4a / E4b / E4c on the pattern path), `ev.idx` otherwise. Neither a location nor a
    moment by itself (user decision 2026-09-24, E2b). Its uses:

    - the event processing order (`processing_order_key`, the sort pins);
    - the historical first-anchor value `SidRecord.creation_event_idx` (stays,
      Plan E E3f) and struct_start's never-established fallback;
    - the H1-overlay window start `export_m15_chart._h1_overlay_window_start_by_sid`
      (the sid's first structural anchor — a location; user decision 2026-09-25).
    Every E2 `# Plan E E3x → moment` TIME half has moved to `event_moment` (E3).
    """
    if ev.type == "CTS_ESTABLISHED":
        return cts_anchor_idx(ev)
    if ev.type == "BOS_CONFIRMED":
        return bos_anchor_idx(ev)
    if ev.type == "CTS_UPDATED":
        return cts_anchor_idx(ev)
    return int(ev.idx)


def processing_order_key(ev: Any) -> Tuple[int, str]:
    """The event processing order: `(stamped_idx, type)` — the pre-E4 `(ev.idx,
    ev.type)`, pinned so the E4 flip reorders nothing (PLAN_E Q3; LANDMINES
    "Event Sort Order Is a Dispatch Invariant"). BOS before EST at a tied index holds by the type
    string ("BOS_CONFIRMED" < "CTS_ESTABLISHED")."""
    return (stamped_idx(ev), ev.type)
