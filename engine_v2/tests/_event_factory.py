"""The one factory for test-built CTS_ESTABLISHED / BOS_CONFIRMED events (Plan E
E2a; PLAN_E §6.1). Keyword-only; parameter names equal the meta keys.

`idx` defaults to what the contract names (ARCHITECTURE "`ev.idx` convention";
`conftest.EVENT_IDX_IS`): the MOMENT `confirmed_at` on CTS_ESTABLISHED (Plan E
E4a), the ANCHOR on BOS_CONFIRMED until E4b flips it here, in one place,
together with `conftest.EVENT_IDX_IS`.

`tests/conftest.py` validates every CTS_ESTABLISHED / BOS_CONFIRMED built during a
test against the same contract, so a hand-built event that skips this factory
still has to be legal.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from engine_v2.structure.market_structure import StructureEvent


def make_cts_established(
    *,
    cts_anchor_idx: int,
    confirmed_at: int,
    price: Optional[float] = None,
    structure_id: Optional[int] = 0,
    cycle_id: Optional[int] = 0,
    struct_direction: Optional[int] = 1,
    pattern_anchor_idx: Optional[int] = None,
    idx: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> StructureEvent:
    """A CTS_ESTABLISHED. `meta` adds extra keys (e.g. `via`); it may not carry
    the keys this factory owns. An attribution argument set to None omits its key."""
    m = _extra(meta, ("cts_anchor_idx", "confirmed_at", "pattern_anchor_idx"))
    if pattern_anchor_idx is not None:
        m["pattern_anchor_idx"] = int(pattern_anchor_idx)
    m["confirmed_at"] = int(confirmed_at)
    _attribution(m, structure_id, cycle_id, struct_direction)
    m["cts_anchor_idx"] = int(cts_anchor_idx)
    return StructureEvent(
        idx=int(confirmed_at if idx is None else idx),
        category="STRUCTURE", type="CTS_ESTABLISHED", price=price, meta=m,
    )


def make_bos_confirmed(
    *,
    bos_anchor_idx: int,
    confirmed_at: int,
    price: Optional[float] = None,
    structure_id: Optional[int] = 0,
    cycle_id: Optional[int] = 0,
    struct_direction: Optional[int] = 1,
    idx: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> StructureEvent:
    """A BOS_CONFIRMED. `meta` adds extra keys (e.g. `source`, `pb_start`)."""
    m = _extra(meta, ("bos_anchor_idx", "confirmed_at"))
    m["confirmed_at"] = int(confirmed_at)
    _attribution(m, structure_id, cycle_id, struct_direction)
    m["bos_anchor_idx"] = int(bos_anchor_idx)
    return StructureEvent(
        idx=int(bos_anchor_idx if idx is None else idx),
        category="STRUCTURE", type="BOS_CONFIRMED", price=price, meta=m,
    )


def make_event(
    etype: str,
    idx: int,
    *,
    price: Optional[float] = None,
    category: str = "STRUCTURE",
    **meta: Any,
) -> StructureEvent:
    """Any structure event; for CTS_ESTABLISHED / BOS_CONFIRMED the `idx`
    ARGUMENT is the ANCHOR (`cts_anchor_idx` / `bos_anchor_idx`) and the event
    goes through the typed factory, whose default sets the emitted `ev.idx`
    (the moment on CTS_ESTABLISHED since Plan E E4a). `meta` must then carry
    `confirmed_at`; `structure_id` / `cycle_id` / `struct_direction` /
    `pattern_anchor_idx` are taken from it."""
    if etype in ("CTS_ESTABLISHED", "BOS_CONFIRMED"):
        m = dict(meta)
        kw: Dict[str, Any] = {
            "confirmed_at": m.pop("confirmed_at"),
            "price": price,
            "structure_id": m.pop("structure_id", None),
            "cycle_id": m.pop("cycle_id", None),
            "struct_direction": m.pop("struct_direction", None),
        }
        assert category == "STRUCTURE"
        if etype == "CTS_ESTABLISHED":
            return make_cts_established(
                cts_anchor_idx=idx, pattern_anchor_idx=m.pop("pattern_anchor_idx", None), meta=m, **kw,
            )
        return make_bos_confirmed(bos_anchor_idx=idx, meta=m, **kw)
    return StructureEvent(idx=int(idx), category=category, type=etype, price=price, meta=dict(meta))


def _extra(meta: Optional[Dict[str, Any]], owned: tuple) -> Dict[str, Any]:
    m = dict(meta or {})
    clash = [k for k in owned if k in m]
    assert not clash, f"pass {clash} as keyword arguments, not in meta"
    return m


def _attribution(m: Dict[str, Any], structure_id, cycle_id, struct_direction) -> None:
    """None omits the key (fixtures that test a missing attribution key)."""
    for k, v in (("cycle_id", cycle_id), ("structure_id", structure_id), ("struct_direction", struct_direction)):
        if v is not None:
            m[k] = int(v)
