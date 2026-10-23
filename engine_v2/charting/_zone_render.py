"""Shared zone-rendering helpers (Item 5, 2026-05-20).

Pure functions consumed by both `export_plotly.py` (main H1 chart) and
`export_m15_chart.py` (sub M15 chart). Computes:

  - Active stretches for KL / POI zones (Spec 3: fill only where the zone
    is active).
  - The stepped-polygon outline path tracing the outer contour of a KL
    zone's `bounds_steps` (so multi-step expansions render as ONE outline,
    not N separate rect outlines).
  - Per-TF tier multipliers for sub-chart zone rendering (Spec 2).
  - The hover text builders shared by both charts (the KL / POI zones, fib, the
    main BOS / CTS dots).

Callers convert (x, y) coordinate lists into Plotly traces; this module
deliberately knows nothing about Plotly so it stays unit-testable.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

from engine_v2.charting.style_registry import STYLE
from engine_v2.zones.kl_zones_v1 import bounds_step_draw_idx
from engine_v2.zones.structure_lifecycle import (
    compute_main_started_by,
    compute_reversal_idx_by_sid,
    compute_struct_start_by_sid,
)
from engine_v2.zones.poi_lifecycle import active_stretches_from_history
from engine_v2.zones.poi_zones import (
    IC_ZONE_FALLBACK,
    IC_ZONE_PRIMARY,
    IP_OUTCOME_NO_PATTERN,
    IP_OUTCOME_PATTERN,
    RANKING_EXCLUDE_PINBAR,
    RANKING_IC_CANDLE_TYPE,
    RANKING_MOST_RECENT_BEFORE_IMBALANCE,
    TIEBREAK_IC_CANDLE_TYPE,
    TIEBREAK_MOST_RECENT_BEFORE_IMBALANCE,
    prior_imbalance_narrowed,
)


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
    window — the same test as `ParentTables`' "collapsed parent cycles",
    in M15 coordinates); on a sub its forming-phase cycles
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

    Delegates the pairing walk to `zones.poi_lifecycle` so the chart fills
    and any other per-candle reader can never disagree (the proximity gate
    and its hover labels read no activation history since PLAN_Q).
    """
    history = zone.meta.get("activation_history", []) or []
    return active_stretches_from_history(history, open_end_idx=render_end_idx)


# ---------------------------------------------------------------------------
# POI hover: the institutional pattern (PLAN_H HV) or the ranking (PLAN_L L2)
# ---------------------------------------------------------------------------

# The ranking rules' runner-up / gap per deciding rule (PLAN_L §4.1, ICR-14): which values the hover line carries.
_RANKING_HOVER_PARTS = {
    RANKING_EXCLUDE_PINBAR: (False, False),
    RANKING_IC_CANDLE_TYPE: (True, False),
    RANKING_MOST_RECENT_BEFORE_IMBALANCE: (True, True),
}


def _only_candidate_text(narrowed: bool) -> str:
    """The ranked IC that entered Step 3 alone (ICR-10: no ranking rule) — `only IC candidate`, or, when Step 1B
    narrowed the field to it (PLAN_S step 3, the user's round 6), `only preferred IC candidate` (it was the only
    PASSER, not the only candidate). ONE wording for the POI hover and the fib hover."""
    return "only preferred IC candidate" if narrowed else "only IC candidate"


def _poi_narrowed(meta: dict) -> bool:
    """Step 1B narrowed this POI's IC candidates (PLAN_S step 3) — read from its meta (`prior_imbalance_narrowed`)."""
    return prior_imbalance_narrowed(meta["prior_imbalance_outcome"], meta["prior_imbalance_passer_count"],
                                    meta["ic_candidate_count"])


def _poi_prior_imbalance_hover_text(meta: dict, ic_idx: int) -> str:
    """Step 1B's POI hover line (PLAN_S step 3; the user's rounds 6–7): when the preference NARROWED the field —
    `<br>ic_prior_imbalance=<c2> unfilled · preferred <passers> of <candidates> IC candidates`, + ` (over pattern
    <rank> at <IC>)` when it outranked a pattern (the IC Step 2 would pick over all the candidates, failing 1B); else
    "" (every candidate passed, or 1B fell back — the user: "Nothing when every candidate passed or it fell back")."""
    outranked_ic, outranked_rank = meta["prior_imbalance_outranked_ic_idx"], meta["prior_imbalance_outranked_ip_rank"]
    assert (outranked_ic is None) == (outranked_rank is None), (
        f"[chart] POI {ic_idx}: outranked IC {outranked_ic} vs rank {outranked_rank!r}")
    if not _poi_narrowed(meta):
        assert outranked_ic is None, f"[chart] POI {ic_idx}: an outranked pattern without a narrowing preference"
        return ""
    c2 = meta["prior_imbalance_first_c2_idx"]
    assert c2 is not None, f"[chart] POI {ic_idx}: a preferred IC without its prior gap"
    text = (f"<br>ic_prior_imbalance={int(c2)} unfilled · preferred {int(meta['prior_imbalance_passer_count'])} of "
            f"{int(meta['ic_candidate_count'])} IC candidates")
    if outranked_ic is not None:
        text += f" (over pattern {outranked_rank} at {int(outranked_ic)})"
    return text


def _poi_ranking_hover_text(meta: dict, ic_idx: int) -> str:
    """A ranked IC's POI hover line (PLAN_L §4.1; PLAN_H §6 ICR-7 / ICR-15): `No pattern · ranking rule <rule>` +
    ` over <runner-up>` + ` (gap <c2>)` when the deciding rule has them (`exclude_pinbar` neither, `ic_candle_type`
    no gap), or `No pattern · only IC candidate` (nothing was ranked — ICR-10; `only preferred IC candidate` when
    Step 1B narrowed the field to it — PLAN_S step 3). NO `IC <idx>`: the POI hover's own `ic_idx=` line names the
    IC. The pattern meta must be all None (asserted)."""
    ic = int(ic_idx)
    pattern = [meta[k] for k in ("ip_rank", "ip_leg_candle_counts", "pattern_anchor_idx", "opp_leg_first_idx",
                                 "pattern_end_idx", "pattern_confirmed_idx", "ip_tiebreak")]
    assert pattern == [None] * len(pattern), f"[chart] ranked POI {ic} carries pattern meta {pattern}"
    rule = meta["ranking_rule"]
    if rule is None:
        assert meta["ranking_runner_up_ic_idx"] is None and meta["ranking_imbalance_first_c2_idx"] is None, (
            f"[chart] POI {ic}: a ranking value without a ranking rule")
        return f"No pattern · {_only_candidate_text(_poi_narrowed(meta))}"
    assert rule in _RANKING_HOVER_PARTS, f"[chart] POI {ic} unknown ranking_rule {rule!r}"
    has_runner_up, has_gap = _RANKING_HOVER_PARTS[rule]
    runner_up, gap = meta["ranking_runner_up_ic_idx"], meta["ranking_imbalance_first_c2_idx"]
    assert (runner_up is not None, gap is not None) == (has_runner_up, has_gap), (
        f"[chart] POI {ic} ranking {rule}: runner-up {runner_up}, gap {gap}")
    text = f"No pattern · ranking rule {rule}"
    if has_runner_up:
        text += f" over {int(runner_up)}"
    if has_gap:
        text += f" (gap {int(gap)})"
    return text


def poi_ic_hover_text(meta: dict, ic_idx: int) -> str:
    """The POI hover's IC lines, every line labelled (the user's zone-hover spec 2026-10-23) — ONE builder for the
    three POI hover sites (the H1 chart, the M15 lens charts' sub POIs and their H1 overlay; PLAN_H HV):
    `ic_reason=IP <rank> · legs a-b-c (initial | opp | continuation)` with each leg as `first–last` (one candle: its
    index), + `<br>ic_tiebreak=over <IC>: IC candle type` / `…: most recent before imbalance <c2>` when a tiebreaker
    decided (`poi_zones.TIEBREAK_*`; the `immediate unfilled imbalance` line went with its rule, PLAN_P). Indices are
    the POI's own (entity-absolute on a mirrored sub POI); the counts must match the leg indices (asserted — a missed
    mirror shift would break it). A ranked IC (`ip_outcome` "no_pattern", PLAN_L L2): its ranking line instead,
    `ic_reason=No pattern · …` (`_poi_ranking_hover_text`). Then Step 1B's `ic_prior_imbalance=` line when the
    preference narrowed the field (`_poi_prior_imbalance_hover_text`, PLAN_S step 3). An IC from the 80–100 fallback
    zone (Step 1A, `ic_zone_band`) adds `<br>ic_zone=80-100 (fallback)`; the 61.8–80 band adds nothing (PLAN_P)."""
    assert meta["ic_zone_band"] in (IC_ZONE_PRIMARY, IC_ZONE_FALLBACK), (
        f"[chart] POI {ic_idx} ic_zone_band {meta['ic_zone_band']!r}")
    tail = _poi_prior_imbalance_hover_text(meta, ic_idx)            # Step 1B's line, then the band's
    tail += "<br>ic_zone=80-100 (fallback)" if meta["ic_zone_band"] == IC_ZONE_FALLBACK else ""
    if meta["ip_outcome"] == IP_OUTCOME_NO_PATTERN:
        return "ic_reason=" + _poi_ranking_hover_text(meta, ic_idx) + tail
    assert meta["ip_outcome"] == IP_OUTCOME_PATTERN, f"[chart] POI {ic_idx} ip_outcome {meta['ip_outcome']!r}"
    assert meta["ranking_rule"] is None, f"[chart] pattern POI {ic_idx} carries ranking_rule {meta['ranking_rule']!r}"
    n_init, n_opp, n_cont = (int(x) for x in meta["ip_leg_candle_counts"].split("-"))
    anchor, opp_first, end = (int(meta["pattern_anchor_idx"]), int(meta["opp_leg_first_idx"]),
                              int(meta["pattern_end_idx"]))
    ic = int(ic_idx)
    assert (opp_first - anchor, ic - opp_first + 1, end - ic) == (n_init, n_opp, n_cont), (
        f"[chart] POI {ic} legs {meta['ip_leg_candle_counts']} vs indices {anchor}/{opp_first}/{end}")

    def _rng(lo: int, hi: int) -> str:
        return f"{lo}" if lo == hi else f"{lo}–{hi}"

    text = (f"ic_reason=IP {meta['ip_rank']} · legs {n_init}-{n_opp}-{n_cont} "
            f"({_rng(anchor, opp_first - 1)} | {_rng(opp_first, ic)} | {_rng(ic + 1, end)})")
    tb = meta["ip_tiebreak"]
    if tb == TIEBREAK_IC_CANDLE_TYPE:
        text += f"<br>ic_tiebreak=over {meta['ip_tb_runner_up_ic_idx']}: IC candle type"
    elif tb == TIEBREAK_MOST_RECENT_BEFORE_IMBALANCE:
        text += (f"<br>ic_tiebreak=over {meta['ip_tb_runner_up_ic_idx']}: "
                 f"most recent before imbalance {meta['ip_tb_imbalance_first_c2_idx']}")
    else:
        assert tb is None, f"[chart] POI {ic} unknown ip_tiebreak {tb!r}"
    return text + tail


# ---------------------------------------------------------------------------
# Fib records: the draw gate + the hover (both charts)
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


def ic_rows_by_fib_record(ic_candidates: Sequence[Any], *, by_sub: bool) -> dict:
    """The `attrs["ic_candidates"]` row holding each fib record's IC outcome —
    its picked row or its no-IC row (`ic_idx` None) — keyed by
    `(structure_id, cycle_id, bos_anchor_idx, cts_anchor_idx)`, prefixed with
    `meta["sub_id"]` when `by_sub` (an M15 lens: a sub's structure_id restarts
    at 0 in every sub). A record the POI record gate skipped has no row."""
    out: dict = {}
    for r in ic_candidates:
        if not (r.picked or r.ic_idx is None):
            continue
        key = (int(r.structure_id), int(r.cycle_id), int(r.bos_anchor_idx), int(r.cts_anchor_idx))
        if by_sub:
            key = (int(r.meta["sub_id"]),) + key
        assert key not in out, f"[chart] two IC outcome rows for fib record {key}"
        out[key] = r
    return out


def fib_hover_limit_text(fib_st: Any) -> Optional[str]:
    """The fib hover's cross-fib LIMIT line (PLAN_I R6-7; the user's spec
    2026-10-23, `plans/qa_chart_edits_inputs/user_spec_20261023_fib_hover.md`)
    from the record's `cross_fib_limit_*` meta — None when the record carries
    none (a fixture). `limit_price=<P> | <source>[ (sub <from>)] |
    cycle_limit=<M | none>` (`M` = the last cycle a cross may target; `none` =
    no cycle's CTS got past `P`: every cycle may cross), or, without `P`,
    `limit_price=NA (<why>)` alone — no cycle may cross, so no `cycle_limit`
    (the user's decision 2): the structure's start (`start_trading`, a
    `first_confluence on parent cycle 0`), `no previous sub`, or
    `prev_sub_no_cts (sub <from>)`. Example (confluence sub 7):
    `limit_price=0.57112 | prev_sub_bos_threshold (sub 6) | cycle_limit=none`. A
    `P` that MOVED by the structure's end (`cross_fib_limit_move_count`, PLAN_M
    CL-6) qualifies the price: `limit_price=0.57630 (moved 7× from 0.57328) |
    parent_bos_threshold | cycle_limit=4`."""
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
        return f"limit_price=NA ({why})"
    M = m["cross_fib_limit_cycle"]
    moved = ""
    if "cross_fib_limit_move_count" in m:
        moved = (f" (moved {int(m['cross_fib_limit_move_count'])}× from "
                 f"{float(m['cross_fib_limit_price_at_start']):.5f})")
    return (f"limit_price={float(price):.5f}{moved} | {source}{sub} | "
            f"cycle_limit={'none' if M is None else int(M)}")


def fib_hover_ic_ref_text(fib_st: Any) -> Optional[str]:
    """The fib hover's IC reference line (PLAN_K A17; RX1-C1 — hover + CSV only; the user's spec 2026-10-23) from the
    record's `ic_ref_*` meta (`FibTracker.stamp_ic_references`): `ic_ref: price=<price> (<source>[, cycle <T−1>]) |
    before=<idx>` — an IC candidate must lie strictly before `before` (the break's pattern anchor), `before=NA (no
    break)` when no break is known (the price side only) — or `ic_ref: NA (no reference)` (the condition passes).
    None for a record the finalize never stamped (no `cross_fib_limit_*` keys — a fixture), like
    `fib_hover_limit_text`. Example (H1 sid 0 cycle 1): `ic_ref: price=0.58246 (prev_cycle_cts, cycle 0) |
    before=651`."""
    m = getattr(fib_st, "meta", None) or {}
    if "cross_fib_limit_source" not in m:
        return None
    if "ic_ref_source" not in m:
        return "ic_ref: NA (no reference)"
    source = m["ic_ref_source"]
    cycle = ""
    if source == "prev_cycle_cts":
        cycle = f", cycle {int(fib_st.cycle_id) - 1}"   # CTS_{T−1}, T = the target cycle — a cross too (PLAN_O)
    brk = m.get("ic_ref_break_pattern_anchor_idx")
    before = "NA (no break)" if brk is None else int(brk)
    return f"ic_ref: price={float(m['ic_ref_price']):.5f} ({source}{cycle}) | before={before}"


def fib_hover_lifecycle_text(fib_st: Any) -> str:
    """The fib's own lifecycle, the zones' form (`zone_lifecycle_line`; the user's spec 2026-10-23 — `life=` until
    then): `lifecycle=[<start>, <end | open>] | end_reason=<reason | open>`; start = `start_idx` (the first
    activation, sticky), `HOVER_NEVER_ACTIVE` when None (never active — FibState)."""
    start = HOVER_NEVER_ACTIVE if fib_st.start_idx is None else int(fib_st.start_idx)
    assert (fib_st.end_idx is None) == (fib_st.end_reason is None), (
        f"[chart] fib sid={fib_st.structure_id} cycle={fib_st.cycle_id} end {fib_st.end_idx} with end_reason "
        f"{fib_st.end_reason!r}")
    if fib_st.end_idx is None:
        return f"lifecycle=[{start}, open] | end_reason=open"
    return f"lifecycle=[{start}, {int(fib_st.end_idx)}] | end_reason={fib_st.end_reason}"


def fib_hover_no_ic_text(ic_row: Optional[Any]) -> Optional[str]:
    """The fib hover's IC line — ONLY on a fib with NO IC, which has no POI to carry the reason (the user's spec
    2026-10-23, decision 1; a pick's IC is on its POI hover — the fib hover dropped `IP <rank> · IC <idx>` / `No
    pattern · IC <idx> · …`): the cycle's no-IC row (`ic_rows_by_fib_record`) → `ic_idx=NA (<reason>)`
    (`no_ic_candidate` / `all_ic_candidates_pinbar`, PLAN_H C1) + ` | ic_zone=80-100 (fallback)` when Step 1A searched
    the 80–100 zone too (`ic_zone_band`, PLAN_P: no IC in either zone); no row → `ic_idx=NA (not evaluated: no POI
    record)` (the POI record gate skipped the record — ended before locking, or no CTS_ESTABLISHED). None on a pick."""
    if ic_row is None:
        return "ic_idx=NA (not evaluated: no POI record)"
    assert ic_row.ic_zone_band in (IC_ZONE_PRIMARY, IC_ZONE_FALLBACK), (
        f"[chart] IC row {ic_row.ic_idx} ic_zone_band {ic_row.ic_zone_band!r}")
    if ic_row.ic_idx is not None:
        return None
    zone = " | ic_zone=80-100 (fallback)" if ic_row.ic_zone_band == IC_ZONE_FALLBACK else ""
    return f"ic_idx=NA ({ic_row.reason}){zone}"


def fib_hovertemplate(tf: str) -> str:
    """The fib hover — ONE template for the H1 chart (`tf="1H"`) and the M15 lens charts (`"15M"`) (PLAN_H §8.9 F1;
    the user's spec 2026-10-23); customdata = `fib_hover_customdata`. Slot 0 = the identity line(s); slot 5 = the
    `ic_ref` + limit lines, each with its own leading `<br>` (absent on an unstamped fixture record); slot 7 = the
    lifecycle line [+ the no-IC line]."""
    return (
        f"TF={tf}<br>"
        "<b>Fib Zone</b><br>"
        "%{customdata[0]}<br>"
        "BOS: idx=%{customdata[1]} price=%{customdata[2]}<br>"
        "CTS: idx=%{customdata[3]} price=%{customdata[4]}"
        "%{customdata[5]}<br>"
        "status=%{customdata[6]}<br>"
        "%{customdata[7]}<br>"
        "61.8%=%{customdata[8]}<br>"
        "80%=%{customdata[9]}<br>"
        "struct_direction=%{customdata[10]}"
        "<extra></extra>"
    )


def fib_hover_customdata(fib_st: Any, ic_row: Optional[Any], price_618: float, price_80: float,
                         id_lines: Optional[str] = None) -> list:
    """`fib_hovertemplate`'s customdata row (slots: 0 the identity line(s) — `id_lines`, a sub's `sub_id | cycle_id`
    + its `first record:` line; None = the main's `sid | cycle_id` —, 1–4 the BOS / CTS anchors, 5 the IC reference
    (`fib_hover_ic_ref_text`) + the cross-fib limit (`fib_hover_limit_text`), 6 the status (+ ` (locked)`), 7 the
    lifecycle (`fib_hover_lifecycle_text`) + the no-IC line (`fib_hover_no_ic_text`; `ic_row` = the cycle's IC
    outcome row, `ic_rows_by_fib_record`), 8–9 the 61.8 / 80 prices, 10 `struct_direction`)."""
    if id_lines is None:
        id_lines = main_zone_id_line({"structure_id": fib_st.structure_id, "cycle_id": fib_st.cycle_id})
    ref_lines = "".join(f"<br>{line}" for line in (fib_hover_ic_ref_text(fib_st), fib_hover_limit_text(fib_st))
                        if line is not None)
    status = getattr(fib_st, "status", "active")
    no_ic = fib_hover_no_ic_text(ic_row)
    lifecycle = fib_hover_lifecycle_text(fib_st) + ("" if no_ic is None else f"<br>{no_ic}")
    return [id_lines, int(fib_st.bos_idx), f"{fib_st.bos_price:.5f}", int(fib_st.cts_idx), f"{fib_st.cts_price:.5f}",
            ref_lines, f"{status}{' (locked)' if fib_st.locked else ''}", lifecycle,
            f"{price_618:.5f}", f"{price_80:.5f}", int(fib_st.struct_direction)]


# ---------------------------------------------------------------------------
# BOS / CTS dot hover — the main structures (the user's spec 2026-10-22,
# plans/qa_chart_edits_inputs/user_spec_20261022_dot_hover.md)
# ---------------------------------------------------------------------------

#: The trigger index of a `start_trading` structure: it has no trigger candle
#: (the backtest picks the start once from the whole window).
HOVER_NA = "NA"

#: The main BOS / CTS / PB dot hover — ONE template for the H1 chart and the
#: M15 charts' H1 overlay; customdata = `main_dot_customdata`.
MAIN_DOT_HOVERTEMPLATE = (
    "TF=1H<br>"
    "idx=%{customdata[0]}<br>"
    "kind=%{customdata[1]}<br>"
    "price=%{customdata[2]:.5f}<br>"
    "sid=%{customdata[3]} | cycle_id=%{customdata[4]}<br>"
    "%{customdata[6]}<br>"
    "%{customdata[7]}<br>"
    "struct_direction=%{customdata[5]}"
    "<extra></extra>"
)


def main_dot_structure_lines(events: Sequence[Any]) -> dict:
    """`{sid: (trigger_line, window_line)}` for the BOS / CTS dots of every MAIN
    structure: `trigger=<started_by> | trigger_idx=<idx>` (`compute_main_started_by`;
    a `reversal` structure's trigger candle = the previous sid's reversal
    confirmation, a `start_trading` one has none → `NA`) and
    `struct_window=[<start>, <end | open>] | end_reason=<reversal | open>` — the
    real-time lifecycle: `compute_struct_start_by_sid` (the start the zones and
    wave candles floor against) to the sid's own reversal."""
    rev = compute_reversal_idx_by_sid(events)
    start = compute_struct_start_by_sid(events, rev)
    out = {}
    for sid, started_by in compute_main_started_by(events).items():
        trigger_idx = rev[sid - 1] if started_by == "reversal" else HOVER_NA
        end = rev.get(sid)
        out[sid] = (
            f"trigger={started_by} | trigger_idx={trigger_idx}",
            f"struct_window=[{start[sid]}, {'open' if end is None else end}] | "
            f"end_reason={'open' if end is None else 'reversal'}",
        )
    return out


def main_dot_customdata(p: Sequence[Any], lines_by_sid: dict) -> list:
    """The customdata row of one main dot for `MAIN_DOT_HOVERTEMPLATE`; `p` = the
    dot tuple `(idx, time, price, kind, sid, cycle_id, struct_direction, …)`,
    `lines_by_sid` = `main_dot_structure_lines`."""
    trigger_line, window_line = lines_by_sid[int(p[4])]
    return [p[0], p[3], p[2], p[4], p[5], p[6], trigger_line, window_line]


# ---------------------------------------------------------------------------
# KL / POI zone hover — main and sub (the user's spec 2026-10-23,
# plans/qa_chart_edits_inputs/user_spec_20261023_zone_hover.md)
# ---------------------------------------------------------------------------

#: A zone that never activated (a POI in a live cycle that never met its activation
#: conditions — none drawn on the reference window): its lifecycle start and its
#: `confirmed_idx`.
HOVER_NEVER_ACTIVE = "NA (never active)"


def zone_lifecycle_line(meta: dict) -> str:
    """The zone's OWN active window (the user 2026-10-23, not its cycle's nor its structure's):
    `lifecycle=[<start>, <end | open>] | end_reason=<reason | open>` — start = the first `active` entry of
    `activation_history` (a KL zone: its `confirmed_idx`; a POI: its FIRST activation, while its `confirmed_idx` is
    the most recent one), end = the cycle's `end_idx` / `end_reason` every zone inherits (`compute_cycle_lifecycle`)."""
    start = next((int(ev["idx"]) for ev in meta["activation_history"] if ev["active"]), HOVER_NEVER_ACTIVE)
    end_idx, end_reason = meta["end_idx"], meta["end_reason"]
    assert (end_idx is None) == (end_reason is None), f"[chart] zone end {end_idx} with end_reason {end_reason!r}"
    if end_idx is None:
        return f"lifecycle=[{start}, open] | end_reason=open"
    return f"lifecycle=[{start}, {int(end_idx)}] | end_reason={end_reason}"


def zone_size_text(top: float, bottom: float, pip_size: float) -> str:
    """`zone_size` = the distance between top and bottom in pips, one decimal: `10.2 pips`."""
    return f"{abs(float(top) - float(bottom)) / float(pip_size):.1f} pips"


def main_zone_id_line(meta: dict) -> str:
    """The main zone hover's identity line: `sid=<structure_id> | cycle_id=<cycle_id>`."""
    return f"sid={int(meta['structure_id'])} | cycle_id={int(meta['cycle_id'])}"


def kl_zone_hovertemplate(tf: str) -> str:
    """The KL zone hover — ONE template for the H1 chart and the M15 charts' H1 overlay (`tf="1H"`) and the sub
    zones (`"15M"`); customdata = `kl_zone_customdata`. Slot 1 = the identity line(s): `main_zone_id_line`, or a
    sub's `sub_id | cycle_id` + its `first record:` line."""
    return (
        f"TF={tf}<br>"
        "KL Zone<br>"
        "side=%{customdata[0]}<br>"
        "%{customdata[1]}<br>"
        "base_pattern=%{customdata[2]}<br>"
        "base_idx=%{customdata[3]}<br>"
        "confirmed_idx=%{customdata[4]}<br>"
        "%{customdata[5]}<br>"
        "top=%{customdata[6]:.5f}<br>"
        "bottom=%{customdata[7]:.5f}<br>"
        "zone_size=%{customdata[8]}<br>"
        "struct_direction=%{customdata[9]}"
        "<extra></extra>"
    )


def kl_zone_customdata(zone: Any, id_lines: str, top: float, bottom: float, pip_size: float) -> list:
    """The customdata row of one KL hover step (`top` / `bottom` = the step's bounds, so `zone_size` is the step's)
    for `kl_zone_hovertemplate`. A KL zone has ONE active stretch, from its `confirmed_idx` (asserted), so its
    lifecycle start repeats `confirmed_idx` — kept for one hover shape with the POI (the user 2026-10-23)."""
    m = zone.meta
    conf = int(m["confirmed_idx"])
    first_active = [int(ev["idx"]) for ev in m["activation_history"] if ev["active"]]
    assert first_active == [conf], f"[chart] KL zone confirmed {conf}, activations {first_active}"
    return [str(zone.side), id_lines, str(m["base_pattern"]), int(m["base_idx"]), conf, zone_lifecycle_line(m),
            float(top), float(bottom), zone_size_text(top, bottom, pip_size), int(m["struct_direction"])]


def poi_zone_hovertemplate(tf: str) -> str:
    """The POI zone hover — ONE template for the H1 chart and the M15 charts' H1 overlay (`tf="1H"`) and the sub
    POIs (`"15M"`); customdata = `poi_zone_customdata`; slot 1 as in `kl_zone_hovertemplate`, slot 4 = the labelled
    IC lines (`poi_ic_hover_text`)."""
    return (
        f"TF={tf}<br>"
        "<b>POI Zone</b><br>"
        "side=%{customdata[0]}<br>"
        "%{customdata[1]}<br>"
        "ic_idx=%{customdata[2]}<br>"
        "confirmed_idx=%{customdata[3]}<br>"
        "%{customdata[4]}<br>"
        "%{customdata[5]}<br>"
        "top=%{customdata[6]:.5f}<br>"
        "bottom=%{customdata[7]:.5f}<br>"
        "zone_size=%{customdata[8]}<br>"
        "struct_direction=%{customdata[9]}"
        "<extra></extra>"
    )


def poi_zone_customdata(zone: Any, id_lines: str, top: float, bottom: float, pip_size: float) -> list:
    """The customdata row of one POI hover for `poi_zone_hovertemplate`. `confirmed_idx` = the MOST RECENT
    activation (`poi_zones`), the lifecycle start = the first; a POI that never activated shows `HOVER_NEVER_ACTIVE`
    for both."""
    m = zone.meta
    conf = HOVER_NEVER_ACTIVE if m["confirmed_idx"] is None else int(m["confirmed_idx"])
    return [str(zone.side), id_lines, int(zone.ic_idx), conf, poi_ic_hover_text(m, zone.ic_idx),
            zone_lifecycle_line(m), float(top), float(bottom), zone_size_text(top, bottom, pip_size),
            int(m["struct_direction"])]


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

    Each step's x is `bounds_step_draw_idx(step)` (the INIT step's `base_idx`, an
    expansion's `expanded_idx` — PLAN_N N1).

    Returns `(xs, ys)` lists ready to feed to `go.Scatter(mode='lines')`.
    """
    # Build per-step effective (x0, x1, top, bottom) tuples, clamped to zone range.
    # (Steps are appended in index order, so the sort is a defence, not a reorder — PLAN_N P6.)
    sorted_steps = sorted(bounds_steps, key=bounds_step_draw_idx)
    effective: List[Tuple[Any, Any, float, float]] = []
    for k, s in enumerate(sorted_steps):
        step_x0 = x_resolver(bounds_step_draw_idx(s))
        if step_x0 is None:
            continue
        if step_x0 < x_start:
            step_x0 = x_start
        if k + 1 < len(sorted_steps):
            nxt_x0 = x_resolver(bounds_step_draw_idx(sorted_steps[k + 1]))
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
