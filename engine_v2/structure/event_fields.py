"""Named reads of a structure event's candle indices (Plan E; ARCHITECTURE
"`ev.idx` convention"; GLOSSARY "Naming Standard").

A CTS / BOS event carries two candle indices with different roles:

- the ANCHOR — the price location adopted as the element's endpoint
  (`cts_anchor_idx`, `bos_anchor_idx`), and the pattern-realm anchor
  (`pattern_anchor_idx`, the breakout pattern's first candle);
- the MOMENT — the candle at which the event became knowable
  (`event_moment`).

Every production read of a `CTS_ESTABLISHED` / `BOS_CONFIRMED` index names its
role through this module, so the E4 flip (`ev.idx` := the moment) changes no
reader. Direct indexing only: every emitter writes these keys (event contract,
LANDMINES "Event Contract Rules" rule 3) — never `.get(key, ev.idx)`.

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
# other `via` is a breakout-pattern name, whose `ev.idx` is the CTS anchor.
CTS_UPDATED_RAW_VIA = "replay_raw"

_CTS_ANCHOR_META_TYPES = ("CTS_ESTABLISHED", "CTS_CONFIRMED", "CTS_RECONFIRMED")
_PATTERN_ANCHOR_TYPES = ("CTS_ESTABLISHED", "REVERSAL_CANDIDATE", "REVERSAL_WATCH_START")


def cts_anchor_idx(ev: Any) -> int:
    """The CTS endpoint (a price location) the event refers to.

    - CTS_ESTABLISHED / CTS_CONFIRMED / CTS_RECONFIRMED: `meta["cts_anchor_idx"]`
      (a CONFIRMED / RECONFIRMED emitted with no current CTS carries None — a
      state the engine never reaches — and raises TypeError here, loudly).
    - CTS_UPDATED: `ev.idx` (both paths; the pattern path until Plan E E4c).

    Any other type raises.
    """
    if ev.type in _CTS_ANCHOR_META_TYPES:
        return int(ev.meta["cts_anchor_idx"])
    if ev.type == "CTS_UPDATED":
        return int(ev.idx)
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

    - CTS_ESTABLISHED / BOS_CONFIRMED: `meta["confirmed_at"]` (until Plan E E4
      `ev.idx` is the anchor).
    - CTS_CONFIRMED / CTS_RECONFIRMED: `ev.idx` (the confirmation candle).
    - CTS_UPDATED: `ev.idx` on the raw path (`via == CTS_UPDATED_RAW_VIA`, the
      processing candle); `meta["confirmed_at"]` on the pattern path (the
      pattern's apply candle, Plan E E3·0 — its `ev.idx` is the CTS anchor
      until E4c).
    - CTS_THRESHOLD_UPDATED: `ev.idx` (the processing candle,
      `_sync_thresholds_from_range`).

    Any other type raises: a new consumer must define its event's moment.
    """
    if ev.type in ("CTS_ESTABLISHED", "BOS_CONFIRMED"):
        return int(ev.meta["confirmed_at"])
    if ev.type in ("CTS_CONFIRMED", "CTS_RECONFIRMED", "CTS_THRESHOLD_UPDATED"):
        return int(ev.idx)
    if ev.type == "CTS_UPDATED":
        if ev.meta["via"] == CTS_UPDATED_RAW_VIA:
            return int(ev.idx)
        return int(ev.meta["confirmed_at"])
    raise ValueError(f"event_moment: no moment defined for {ev.type}")


def stamped_idx(ev: Any) -> int:
    """The index `ev.idx` holds TODAY, frozen against the Plan E E4 flip: the
    anchor for CTS_ESTABLISHED / BOS_CONFIRMED / CTS_UPDATED, `ev.idx`
    otherwise. Neither a location nor a moment by itself (user decision
    2026-09-24, E2b). Two uses only:

    - the event processing order (`processing_order_key`, the sort pins);
    - an E2 TIME half over mixed event types (CTS lists, the all-type
      struct_start minima, the threshold timeline) that an E3 stage will switch
      — always written with its `# Plan E E3x → moment` marker. Where the list
      holds types `event_moment` does not define, that stage must define them
      first (PLAN_E §7, E3f / E3g-2 notes).
    """
    if ev.type == "CTS_ESTABLISHED":
        return cts_anchor_idx(ev)
    if ev.type == "BOS_CONFIRMED":
        return bos_anchor_idx(ev)
    if ev.type == "CTS_UPDATED":
        return cts_anchor_idx(ev)
    return int(ev.idx)


def processing_order_key(ev: Any) -> Tuple[int, str]:
    """The event processing order: `(stamped_idx, type)` — today's `(ev.idx,
    ev.type)`, pinned so the E4 flip reorders nothing (PLAN_E Q3; LANDMINES
    "Event Sort Order Is a Dispatch Invariant"). BOS before EST at a tied index holds by the type
    string ("BOS_CONFIRMED" < "CTS_ESTABLISHED")."""
    return (stamped_idx(ev), ev.type)
