# engine_v2/patterns/institutional_pattern.py
"""
Institutional pattern (IP) — the leg classification of ONE IC candidate.

Canonical reading: `engine_v2/plans/PLAN_H_institutional_pattern_poi.md` §1 items 2–10 (the user's spec:
`plans/plan_h_inputs/institutional_pattern_spec.md`; every decision in PLAN_H §6). This module is the PATTERN only —
no fib, no candidate rules: `zones.poi_zones.select_ic` applies the IC conditions (POI_ZONES_SPEC §3 Step 1 — §3.1,
the 61.8–80 touch among them) first and this last (D4).

Around an IC (an agreed −sd candle), in struct direction `sd`:

  initial leg (agreed sd) · opp leg (agreed −sd, the IC its LAST candle) · continuation leg (agreed sd)

- Legs read the AGREED direction (`features.candles_v2.agreed_direction`): a split candle breaks every run; a true
  doji (0) never breaks a run and is a full member of the leg it is assigned to (cap + multi); a volume-only doji is
  an ordinary directional candle.
- The IC is the last opp candle: after it (and any true dojis, which join the continuation) the next candle must be
  agreed sd.
- Runs are the maximal consecutive runs, cut at the fib's BOS / CTS anchors (the anchor candles are members, B1a);
  the continuation = the FIRST `cap` candles of its run, the initial = the MOST RECENT `cap` of its run; the opp run
  is never cut (P5).
- True dojis between the initial and opp runs (the junction) are tried in each leg — every split (J1); placing one in
  the initial leg at the cap slides the window (its earliest candle drops); a full tie → rule A / B (fill the initial
  leg up to the cap, the rest to opp).
- E1: a true doji never starts or ends the pattern — it is trimmed off the initial leg's first and the continuation
  leg's last position (also where the cap cuts a run there).
- D3c: next to the cap-`cap` initial, the single candle right before the opp leg is evaluated as a 3D initial (cap 1,
  accepting 3D only); a true doji is never a 3D initial.
- Stacked legs are classified with the SAME classifier and params as single candles (P2): one `classify_candles` call
  for every stack of every candidate (`classify_ics`); big flags via `candles_v2.big_ratio_flags` against
  `prior_maru_max_len_as0` at the variant's `pattern_anchor_idx` (P3), gated by type (big_maru needs a maru,
  big_normal a normal — N2).
- Ranks 1A–3D (`SINGLE_OPP_RANK` / `MULTI_OPP_RANK`, then the shape i–iv); the continuation must be big_maru with a
  stacked body strictly greater than the opp leg's.

Indices: the frame must carry a 0-based contiguous index (labels == positions — every engine frame does: the main df,
the reset sub slices, the probe's `iloc` prefixes); asserted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from engine_v2.features.candle_params import CandleParams
from engine_v2.features.candles_v2 import big_ratio_flags, classify_candles, compute_candle_metrics

# Rank order (best first) and shapes. A shape = (multi initial, multi continuation): i single/single, ii
# multi/single, iii single/multi, iv multi/multi. 3D (a single initial only) has i (single continuation) and ii
# (multi continuation).
RANKS: Tuple[str, ...] = ("1A", "1B", "2A", "2B", "2C", "3A", "3B", "3C", "3D")
SHAPES: Tuple[str, ...] = ("i", "ii", "iii", "iv")
_SHAPE = {(False, False): "i", (True, False): "ii", (False, True): "iii", (True, True): "iv"}

# The rank table — ONE data constant (a chart-review re-rank is a one-line change). Keys: the initial stack's kind
# ("big_maru" / "maru" / "big_normal") and, for a SINGLE opp leg, the IC's candle type.
SINGLE_OPP_RANK: Dict[Tuple[str, str], str] = {
    ("big_maru", "maru"): "1A", ("big_maru", "normal"): "2A", ("big_maru", "pinbar"): "3A",
    ("maru", "maru"): "1B", ("maru", "normal"): "2A", ("maru", "pinbar"): "3A",
    ("big_normal", "maru"): "2B", ("big_normal", "normal"): "2C", ("big_normal", "pinbar"): "3B",
}
# A MULTI opp leg (no type requirement on it — P8a): 3C from any maru initial; 3D from a single big_normal initial.
MULTI_OPP_RANK: Dict[str, str] = {"big_maru": "3C", "maru": "3C", "big_normal": "3D"}

# The IC's candle type, best first — read by the shared candle-type rule `poi_zones.prefer_candle_type` (the tiebreak
# rules' `TIEBREAK_IC_CANDLE_TYPE` with the pinbar; the ranking rules, PLAN_L L2, without it).
IC_TYPE_PREFERENCE: Tuple[str, ...] = ("maru", "normal", "pinbar")


@dataclass(frozen=True)
class IPResult:
    """One IC candidate's institutional pattern (or why it has none).

    `rank` / `shape` None ⇔ no pattern; `reason` then names the first failing rule, `ip_leg_candle_counts` /
    `detail` the first failing variant's (when one was ranked). The legs are contiguous around the IC (initial
    `ic - opp - initial + 1 ..`, opp `.. ic`, continuation `ic + 1 ..`), so the counts locate them; for a pattern
    `pattern_anchor_idx` (the initial leg's first candle), `opp_leg_first_idx`, `ic_idx` (the opp leg's last),
    `pattern_end_idx` (the continuation's last) are set too. `detail` holds no index (the sub mirror shifts only
    the index fields).
    `pattern_confirmed_idx` (R1, a MOMENT) = when the continuation leg is known complete: its last candle when the
    run (before the E1 trim) reached the cap, else the next candle — the one that shows the leg ended (also for a run
    cut at the CTS anchor); may lie past the frame (then not yet known)."""
    ic_idx: int
    rank: Optional[str] = None
    shape: Optional[str] = None
    pattern_anchor_idx: Optional[int] = None
    opp_leg_first_idx: Optional[int] = None
    pattern_end_idx: Optional[int] = None
    pattern_confirmed_idx: Optional[int] = None
    ip_leg_candle_counts: Optional[str] = None      # "a-b-c" (initial-opp-continuation)
    reason: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def ip_rank(self) -> Optional[str]:
        """The rank label, e.g. "2A.i" (PLAN_H §4)."""
        return f"{self.rank}.{self.shape}" if self.rank is not None else None

    @property
    def order_key(self) -> Tuple[int, int]:
        """Best first: the rank letter, then the shape (R1 — PLAN_H §1 item 10)."""
        assert self.rank is not None, f"IC {self.ic_idx} has no pattern"
        return RANKS.index(self.rank), SHAPES.index(self.shape)

    def __deepcopy__(self, memo):
        """Immutable after construction — shared, never copied (pandas deep-copies `df.attrs` on every
        `__finalize__`; the `ImbalanceInstance.__deepcopy__` reasoning, GOTCHAS)."""
        return self


@dataclass(frozen=True)
class LegVariant:
    """One leg assignment of a candidate (positions): a junction split × the cap-`cap` / the D3c cap-1 initial."""
    initial: Tuple[int, ...]
    opp: Tuple[int, ...]
    continuation: Tuple[int, ...]
    placement: int          # junction true dojis given to the initial leg
    only_3d: bool           # the D3c cap-1 initial: accepts 3D only


def _check_index(df: pd.DataFrame) -> None:
    assert df.index.equals(pd.RangeIndex(len(df))), (
        "[institutional_pattern] the frame's index must be 0-based contiguous (labels == positions)")


def _trim(leg: List[int], agreed: np.ndarray, *, lead: bool) -> List[int]:
    """E1: drop true dojis off the leg's OUTER end (lead: its start — the pattern's first candle; else its end)."""
    leg = list(leg)
    while leg and agreed[leg[0 if lead else -1]] == 0:
        leg.pop(0 if lead else -1)
    return leg


def leg_variants(
    agreed: np.ndarray,
    ic: int,
    sd: int,
    *,
    bos_anchor_idx: int,
    cts_anchor_idx: int,
    cap: int,
) -> Tuple[Optional[List[LegVariant]], int, Optional[int], Optional[str]]:
    """Every leg assignment of the IC at position `ic` → `(variants, rule_ab_placement, pattern_confirmed_idx,
    reason)`; `variants` None (with `reason`) when no assignment exists. The scans stay inside
    `[bos_anchor_idx, cts_anchor_idx]` except the opp run's backward scan (an opp run reaching past the BOS anchor
    leaves no initial leg)."""
    # The IC is the LAST opp candle: after it (true dojis join the continuation) the next candle is agreed sd.
    j = ic + 1
    while j <= cts_anchor_idx and agreed[j] == 0:
        j += 1
    if j > cts_anchor_idx or agreed[j] != sd:
        reason = "ic_not_last_opp" if j <= cts_anchor_idx and agreed[j] == -sd else "no_continuation"
        return None, 0, None, reason

    # The continuation run (cut at the CTS anchor), its first `cap` candles, E1-trimmed.
    run: List[int] = []
    j = ic + 1
    while j <= cts_anchor_idx and agreed[j] in (0, sd):
        run.append(j)
        j += 1
    capped = run[:cap]
    confirmed = capped[-1] if len(capped) == cap else capped[-1] + 1
    cont = _trim(capped, agreed, lead=False)
    if not any(agreed[x] == sd for x in cont):
        return None, 0, confirmed, "continuation_no_sd_candle"

    # The whole opp run (never cut, P5); its leading true dojis = the junction.
    opp: List[int] = [ic]
    j = ic - 1
    while j >= 0 and agreed[j] in (0, -sd):
        opp.insert(0, j)
        j -= 1
    junction: List[int] = []
    while len(opp) > 1 and agreed[opp[0]] == 0:
        junction.append(opp.pop(0))

    # The initial run, back from the candle before the opp run (cut at the BOS anchor).
    k = j
    if k < 0 or agreed[k] != sd:
        return None, 0, confirmed, "no_initial_candle"
    full: List[int] = []
    j = k
    while j >= bos_anchor_idx and agreed[j] in (0, sd):
        full.insert(0, j)
        j -= 1
    if not full:
        return None, 0, confirmed, "initial_before_bos_anchor"

    # Rule A / B (a full tie between placements): fill the initial leg up to the cap.
    rule_ab = min(max(cap - len(full), 0), len(junction))
    variants: List[LegVariant] = []
    for placement in range(len(junction) + 1):
        seq = full + junction[:placement]
        opp_s = tuple(junction[placement:] + opp)
        inits = [(_trim(seq[-cap:], agreed, lead=True), False)]
        if len(seq) > 1:
            inits.append((seq[-1:], True))                       # D3c: the cap-1 3D initial
        for init, only_3d in inits:
            if not any(agreed[x] == sd for x in init):           # ≥ 1 sd candle (never a doji-only / doji 3D initial)
                continue
            variants.append(LegVariant(tuple(init), opp_s, tuple(cont), placement, only_3d))
    return variants, rule_ab, confirmed, None


class _Frame:
    """The arrays one classification reads (positions)."""

    def __init__(self, df: pd.DataFrame, params: CandleParams):
        _check_index(df)
        self.o = df["o"].to_numpy(dtype=float)
        self.h = df["h"].to_numpy(dtype=float)
        self.l = df["l"].to_numpy(dtype=float)
        self.c = df["c"].to_numpy(dtype=float)
        self.ctype = df["candle_type"].to_numpy()
        self.ref = df["prior_maru_max_len_as0"].to_numpy(dtype=float)
        self.pair = df.attrs["pair"]            # the frame's own pip size (a KeyError when unstamped — P2)
        self.pip = 0.01 if "JPY" in str(self.pair).upper() else 0.0001
        self.params = params

    def ohlc(self, first: int, last: int) -> Tuple[float, float, float, float]:
        return (self.o[first], float(self.h[first:last + 1].max()), float(self.l[first:last + 1].min()),
                self.c[last])


def _stack_types(fr: _Frame, ranges: Sequence[Tuple[int, int]]) -> Dict[Tuple[int, int], str]:
    """Every multi-candle stack's candle type in ONE classifier call (the frame's params and pair — P2)."""
    multi = sorted({r for r in ranges if r[1] > r[0]})
    if not multi:
        return {}
    rows = [fr.ohlc(a, b) for a, b in multi]
    stacks = pd.DataFrame(rows, columns=["o", "h", "l", "c"])
    stacks.attrs["pair"] = fr.pair
    types = classify_candles(compute_candle_metrics(stacks), fr.params)["candle_type"].to_numpy()
    return {r: str(t) for r, t in zip(multi, types)}


def _stack(fr: _Frame, leg: Tuple[int, ...], types: Dict[Tuple[int, int], str], ref: float) -> Dict[str, Any]:
    first, last = leg[0], leg[-1]
    o, h, l, c = fr.ohlc(first, last)
    typ = str(fr.ctype[first]) if first == last else types[(first, last)]
    flags = big_ratio_flags(h - l, ref, fr.params)
    return {
        "type": typ,
        "ratio": None if flags is None else float(flags[0]),
        "body": abs(c - o),
        "big_maru": typ == "maru" and flags is not None and bool(flags[1]),
        "big_normal": typ == "normal" and flags is not None and bool(flags[2]),
    }


def _initial_kind(a: Dict[str, Any]) -> Optional[str]:
    if a["big_maru"]:
        return "big_maru"
    if a["type"] == "maru":
        return "maru"
    if a["big_normal"]:
        return "big_normal"
    return None


def rank_variant(
    fr: _Frame,
    v: LegVariant,
    ic: int,
    types: Dict[Tuple[int, int], str],
) -> Tuple[Optional[Tuple[str, str]], Optional[str], Dict[str, Any]]:
    """`((rank, shape), None, detail)` or `(None, reason, detail)` for one leg assignment."""
    ref = float(fr.ref[v.initial[0]])                    # the 5 marus before the pattern's first candle (P3)
    a = _stack(fr, v.initial, types, ref)
    m = _stack(fr, v.opp, types, ref)
    e = _stack(fr, v.continuation, types, ref)
    detail: Dict[str, Any] = {}
    for name, st in (("initial", a), ("opp", m), ("continuation", e)):
        detail[f"{name}_type"] = st["type"]
        detail[f"{name}_ratio"] = st["ratio"]
        detail[f"{name}_body_pips"] = round(float(st["body"] / fr.pip), 1)
    if not e["big_maru"]:
        return None, "continuation_not_big_maru", detail
    if not e["body"] > m["body"]:
        return None, "continuation_body_not_greater", detail
    multi_init, multi_cont = len(v.initial) > 1, len(v.continuation) > 1
    kind = _initial_kind(a)
    if len(v.opp) > 1:
        rank = MULTI_OPP_RANK.get(kind) if kind is not None else None
        if rank == "3C" and not v.only_3d:
            return ("3C", _SHAPE[(multi_init, multi_cont)]), None, detail
        if rank == "3D" and not multi_init:
            return ("3D", "ii" if multi_cont else "i"), None, detail
        return None, "initial_fails_multi_opp", detail
    if v.only_3d:
        return None, "3d_needs_multi_opp", detail
    if kind is None:
        return None, "initial_fails", detail
    return (SINGLE_OPP_RANK[(kind, str(fr.ctype[ic]))], _SHAPE[(multi_init, multi_cont)]), None, detail


def classify_ics(
    df: pd.DataFrame,
    ic_idxs: Sequence[int],
    sd: int,
    *,
    bos_anchor_idx: int,
    cts_anchor_idx: int,
    agreed: np.ndarray,
    cap: int,
    params: CandleParams,
) -> List[IPResult]:
    """The institutional pattern of each IC in `ic_idxs` (one `IPResult` each, same order). Every candidate's stacks
    are classified in ONE classifier call. For each IC the best (rank, shape) over its leg variants wins; a full tie
    keeps rule A / B's placement, else the first variant in order (placements 0.., the cap-`cap` initial before the
    D3c cap-1)."""
    fr = _Frame(df, params)
    agreed = np.asarray(agreed)
    per_ic = []
    ranges: List[Tuple[int, int]] = []
    for ic in ic_idxs:
        variants, rule_ab, confirmed, reason = leg_variants(
            agreed, int(ic), int(sd), bos_anchor_idx=int(bos_anchor_idx), cts_anchor_idx=int(cts_anchor_idx),
            cap=int(cap),
        )
        per_ic.append((int(ic), variants, rule_ab, confirmed, reason))
        for v in variants or ():
            ranges += [(v.initial[0], v.initial[-1]), (v.opp[0], v.opp[-1]),
                       (v.continuation[0], v.continuation[-1])]
    types = _stack_types(fr, ranges)

    out: List[IPResult] = []
    for ic, variants, rule_ab, confirmed, reason in per_ic:
        if variants is None:
            out.append(IPResult(ic_idx=ic, pattern_confirmed_idx=None, reason=reason))
            continue
        best = None                     # (key, variant, (rank, shape), detail)
        first_fail = None               # (reason, variant, detail)
        for v in variants:
            res, why, detail = rank_variant(fr, v, ic, types)
            if res is None:
                first_fail = first_fail or (why, v, detail)
                continue
            key = (RANKS.index(res[0]), SHAPES.index(res[1]))
            if best is None or key < best[0] or (key == best[0] and v.placement == rule_ab
                                                 and best[1].placement != rule_ab):
                best = (key, v, res, detail)
        if best is None:
            why, v, detail = first_fail
            out.append(IPResult(ic_idx=ic, reason=why, detail=detail,
                                ip_leg_candle_counts=f"{len(v.initial)}-{len(v.opp)}-{len(v.continuation)}"))
            continue
        _, v, (rank, shape), detail = best
        out.append(IPResult(
            ic_idx=ic,
            rank=rank,
            shape=shape,
            pattern_anchor_idx=int(v.initial[0]),
            opp_leg_first_idx=int(v.opp[0]),
            pattern_end_idx=int(v.continuation[-1]),
            pattern_confirmed_idx=int(confirmed),
            ip_leg_candle_counts=f"{len(v.initial)}-{len(v.opp)}-{len(v.continuation)}",
            detail=detail,
        ))
    return out


def classify_ip(
    df: pd.DataFrame,
    ic_idx: int,
    sd: int,
    *,
    bos_anchor_idx: int,
    cts_anchor_idx: int,
    agreed: np.ndarray,
    cap: int,
    params: CandleParams,
) -> IPResult:
    """`classify_ics` for one IC."""
    return classify_ics(df, [ic_idx], sd, bos_anchor_idx=bos_anchor_idx, cts_anchor_idx=cts_anchor_idx,
                        agreed=agreed, cap=cap, params=params)[0]
