"""Unit + light-integration tests for the unified-probe start resolvers under
Plan C (`plans/PLAN_C_lifecycle_rewrite.md` §5; API contract
`PLAN_C_API_CONTRACT.md`). Written TESTS-FIRST: every class except
`TestMapCandleToLowerTf` / `TestSiblingCtsIdxWindow` targets the Plan-C API and
is expected to fail on the pre-Plan-C base (ImportError / TypeError on the new
names), not on a fixture bug.

Two resolver paths, dispatched by use_case in `_resolve_trigger_m15_start`
(the legacy escape hatch is GONE — §7):
  - `first_confluence` → `_resolve_first_confluence_via_unified_probe`: own
    ad-hoc BOS_0 ref, parent-BOS-extreme price-mapped input, Phase 1+2.
  - `first_counter` / `subsequent_confluence` / `subsequent_counter` →
    `_resolve_sibling_cts_via_unified_probe`: input AND reference co-sourced
    from the SIBLING lens's most recent qualifying CTS, read from the POOL
    (`_build_sibling_cts_ref_zone_from_pool`, §5.1) inside `[lo, hi]` with
    `hi` = the reading trigger's `trigger_idx`; Phase 1 only.

Both return a `ResolvedStart` on success and a `ProbeFailure` on every failure
branch (§4.3 step 2 / contract). The probe cache (§5.3) wraps the
`unified_probe` call inside the resolvers, keyed
`(parent_path, "M15", direction, initial_input_idx)`.

Covers:
  - Refactored `map_candle_to_lower_tf` (unchanged by Plan C; kept verbatim).
  - first_confluence resolver wiring (ResolvedStart / ProbeFailure branches,
    `probe_end_idx` keyword, `ProbeResult.starting_idx`).
  - sibling-CTS resolver wiring against a STUB POOL (records + slice-local
    geometry): lens selection, entity-absolute shift, per-record live-window
    CLIP (§5.1 concrete 3819 handover case), record-level exclusions,
    fallback, degenerate window, `kl_zones=[]` + shared frame.
  - `_sibling_cts_idx_window` (unchanged semantics, `hi` passed in).
  - dispatcher routing + pool/hi threading; legacy hatch absent.
  - probe cache hit / APPROX hit / REF-ZONE DIFFERS / convergence / determinism.
  - §5.2 `finalize_idx == hi` assert for sibling types (cache hit exempt).

The unified-probe primitive itself is exercised in test_unified_probe.py.

Fixture facts used in every derivation below (`_h1_df_uptrend(10)` +
`_m15_from_h1`): H1 idx h ↔ M15 idx [4h, 4h+3]; M15 4h carries the H1 HIGH and
M15 4h+1 the H1 LOW, so price-map(h, +1) = 4h and price-map(h, -1) = 4h+1;
LOH(h) = 4h+3 (`_map_parent_idx_to_m15_hour_end`).
"""
from __future__ import annotations

import contextlib
import dataclasses
import importlib
import inspect
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from engine_v2.multitf.data_bridge import map_candle_to_lower_tf
from engine_v2.multitf import entity_df_mutation as edm
from engine_v2.multitf.entity_df_mutation import (
    ProbeFailure,
    ResolvedStart,
    _build_sibling_cts_ref_zone_from_pool,
    _resolve_first_confluence_via_unified_probe,
    _resolve_sibling_cts_via_unified_probe,
    _resolve_trigger_m15_start,
    _sibling_cts_idx_window,
)
from engine_v2.multitf.sub_structure_pool import (
    ProbeCacheEntry,
    StructureKey,
    SubStructurePool,
    TriggerRecord,
)
from engine_v2.multitf.types import MultiTFTrigger
from engine_v2.structure import reference_zone as rz_mod
from engine_v2.structure import unified_probe as up_mod
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.structure.reference_zone import ReferenceZone
from engine_v2.structure.unified_probe import ProbeResult


# Real signatures captured BEFORE any patching, so call-arg inspection binds
# positional-or-keyword calls uniformly.
_FC_REF_SIG = inspect.signature(edm._build_first_confluence_ref_zone)
_PRIMITIVE_SIG = inspect.signature(rz_mod.build_reference_zone_from_cts_event)

_PARENT_PATH = "H1.main"
_SUB_TF = "M15"


# ---------------------------------------------------------------------------
# Fixtures — synthetic H1 + derived M15 (4 M15 per H1).
# ---------------------------------------------------------------------------

def _h1_df_uptrend(n_hours: int = 20, base_price: float = 0.6000) -> pd.DataFrame:
    base_time = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    rows = []
    price = base_price
    for i in range(n_hours):
        o = price
        c = price + 0.0030
        h = c + 0.0005
        l = o - 0.0005
        rows.append({
            "time": base_time + pd.Timedelta(hours=i),
            "o": round(o, 5), "h": round(h, 5),
            "l": round(l, 5), "c": round(c, 5),
            "volume": 100,
        })
        price = c
    df = pd.DataFrame(rows)
    df.attrs["pair"] = "NZD_USD"
    return df


def _m15_from_h1(h1: pd.DataFrame) -> pd.DataFrame:
    """4 M15 per H1 hour. M15 #0 carries the H1 high, M15 #1 carries the H1 low."""
    rows = []
    for _, hr in h1.iterrows():
        ht = hr["time"]
        ho, hh, hl, hc = float(hr["o"]), float(hr["h"]), float(hr["l"]), float(hr["c"])
        rows.append({
            "time": ht + pd.Timedelta(minutes=0),
            "o": ho, "h": hh, "l": round(ho - 0.0002, 5),
            "c": round(ho + 0.0010, 5), "volume": 25,
        })
        rows.append({
            "time": ht + pd.Timedelta(minutes=15),
            "o": round(ho + 0.0010, 5), "h": round(ho + 0.0015, 5),
            "l": hl, "c": round(ho + 0.0008, 5), "volume": 25,
        })
        rows.append({
            "time": ht + pd.Timedelta(minutes=30),
            "o": round(ho + 0.0008, 5), "h": round(ho + 0.0018, 5),
            "l": round(ho + 0.0005, 5), "c": round(ho + 0.0020, 5),
            "volume": 25,
        })
        rows.append({
            "time": ht + pd.Timedelta(minutes=45),
            "o": round(ho + 0.0020, 5), "h": round(ho + 0.0028, 5),
            "l": round(ho + 0.0018, 5), "c": hc, "volume": 25,
        })
    df = pd.DataFrame(rows)
    df.attrs["pair"] = "NZD_USD"
    return df


@pytest.fixture(scope="module")
def big_frames():
    """Entity-absolute frames large enough to hold the §5.1 concrete case's
    real idxs (3621 / 3819 / 3850 / 3900 / 4083): 1030 H1 hours → 4120 M15
    candles. LOH(926) = 3707, LOH(1020) = 4083 (verified)."""
    h1 = _h1_df_uptrend(n_hours=1030)
    m15 = _m15_from_h1(h1)
    return h1, m15


# ---------------------------------------------------------------------------
# map_candle_to_lower_tf — post-refactor signature
# ---------------------------------------------------------------------------

class TestMapCandleToLowerTf:
    def test_extreme_dir_plus_one_picks_max_high(self):
        h1 = _h1_df_uptrend(n_hours=3)
        m15 = _m15_from_h1(h1)
        result = map_candle_to_lower_tf(
            h1.loc[1, "time"], parent_extreme_dir=1, m15_df=m15,
        )
        assert result == 4  # H1 hour 1 → M15 #0

    def test_extreme_dir_minus_one_picks_min_low(self):
        h1 = _h1_df_uptrend(n_hours=3)
        m15 = _m15_from_h1(h1)
        result = map_candle_to_lower_tf(
            h1.loc[1, "time"], parent_extreme_dir=-1, m15_df=m15,
        )
        assert result == 5  # H1 hour 1 → M15 #1

    def test_empty_hour_returns_none(self):
        m15 = _m15_from_h1(_h1_df_uptrend(n_hours=3))
        far_future = pd.Timestamp("2030-01-01 00:00", tz="UTC")
        assert map_candle_to_lower_tf(far_future, 1, m15) is None

    def test_signature_no_h1_extreme_price_arg(self):
        m15 = _m15_from_h1(_h1_df_uptrend(n_hours=2))
        with pytest.raises(TypeError):
            map_candle_to_lower_tf(  # type: ignore[call-arg]
                m15.loc[0, "time"], 0.6005, 1, m15,
            )


# ---------------------------------------------------------------------------
# Helpers — triggers, fake probe results, stub pool (records + geometry)
# ---------------------------------------------------------------------------

def _make_trigger(
    use_case: str, parent_sid: int = 0, parent_cycle_id: int = 0,
    parent_sd: int = 1, lower_sd: int = -1,
    parent_input_idx: int = 1, probe_end_idx: int = 5,
    prior_sd_trigger_idx: Optional[int] = None,
    prior_cts_prox_idx: Optional[int] = None,
) -> MultiTFTrigger:
    meta = {
        "parent_input_idx": parent_input_idx,
        "probe_end_idx": probe_end_idx,
        "trigger_event_idx": probe_end_idx,
    }
    if prior_sd_trigger_idx is not None:
        meta["prior_sd_trigger_idx"] = prior_sd_trigger_idx
    if prior_cts_prox_idx is not None:
        meta["prior_cts_prox_idx"] = prior_cts_prox_idx
    kwargs = dict(
        parent_tf="H1",
        parent_sid=parent_sid,
        parent_cycle_id=parent_cycle_id,
        parent_sd=parent_sd,
        use_case=use_case,
        lower_tf="M15",
        lower_sd=lower_sd,
        meta=meta,
    )
    return MultiTFTrigger(**kwargs)


def _ref_zone(
    source: str = "ad_hoc_bos_0", anchor_idx: int = 0,
    inner: float = 0.6020, outer: float = 0.6010, side: str = "buy",
) -> ReferenceZone:
    return ReferenceZone(
        outer=outer, inner=inner, side=side,  # type: ignore[arg-type]
        source=source, anchor_idx=anchor_idx,  # type: ignore[arg-type]
    )


def _fake_probe_result(
    starting_idx: int = 4, source: str = "ad_hoc_bos_0", *,
    finalize_idx: Optional[int] = None,
    finalize_condition: str = "no_retrace",
    bos0_inner: Optional[float] = 0.6020,
    status: str = "finalized",
) -> ProbeResult:
    """`ProbeResult` under the §7 rename: `start_idx` → `starting_idx`."""
    return ProbeResult(
        starting_idx=starting_idx, status=status, iterations=1,  # type: ignore[call-arg]
        original_ref_zone=_ref_zone(source=source),
        finalize_condition=finalize_condition,  # type: ignore[arg-type]
        bos0_inner=bos0_inner,
        finalize_idx=finalize_idx,
    )


def _cts_event(
    ev_type: str, idx_local: int, anchor_local: Optional[int] = None,
    *, cycle_id: int = 0, struct_direction: int = 1,
) -> StructureEvent:
    """A SLICE-LOCAL CTS event exactly as a sub's own MS run emits it
    (structure_id=0 local; NO parent_sid / parent_cycle_id / sub_sid stamps —
    the pool geometry is raw MS output, §2.2)."""
    meta = {
        "structure_id": 0,
        "cycle_id": cycle_id,
        "struct_direction": struct_direction,
        "confirmed_at": idx_local,
    }
    if ev_type == "CTS_CONFIRMED":
        meta["cts_anchor_idx"] = anchor_local
    if ev_type == "CTS_UPDATED":
        meta["via"] = "continuous"   # a pattern-path update: confirmed_at = its apply (== idx here)
        meta["cts_anchor_idx"] = idx_local if anchor_local is None else anchor_local  # Plan E E4c
    return StructureEvent(
        idx=idx_local, category="STRUCTURE", type=ev_type,
        price=0.6020, meta=meta,
    )


def _stub_sub(pool: SubStructurePool, *, direction: int, starting_idx: int,
              slice_begin: int, events):
    """A pool entry with stub geometry `(bounded, slice_begin)`; `bounded`
    carries SLICE-LOCAL events (§2.2), df=None (the sibling read must use the
    shared m15 frame, not the geometry's df — §5.1)."""
    sub, created = pool.get_or_create(
        StructureKey(_PARENT_PATH, _SUB_TF, int(direction), int(starting_idx)),
    )
    assert created
    sub.geometry = (SimpleNamespace(events=list(events), df=None), int(slice_begin))
    return sub


def _rec(sub, *, lens: str, S: int, C: int, tss: int, trigger_type: str,
         trigger_idx: int, start_idx: int, finalize_idx: Optional[int] = None,
         trigger_end_idx: Optional[int] = None, end_reason: Optional[str] = None,
         ended_by: Optional[int] = None, seq: int = 0, pool=None,
         relative_dir: str = "confluence", parent_floor_idx: int = 0) -> TriggerRecord:
    """Register a §2.1 `TriggerRecord` on `sub` so `pool.records_for(...)` sees
    it. Plan C §4.3 step 7 registers a record as `sub.records.append(rec)`;
    if the pool exposes an `add_record` registrar (an index the contract does
    not list), that is used instead so the record is visible either way.
    `end_idx = max(trigger_end_idx, start_idx)` per §2.1; `probe_finalize_idx`
    defaults to `trigger_idx` (non-FC types: finalize == trigger, §5.2)."""
    rec = TriggerRecord(
        lens=lens, parent_sid=S, parent_cycle_id=C, trigger_sub_sid=tss,
        sub_id=sub.sub_id,
        trigger_type=trigger_type, trigger_idx=trigger_idx,
        probe_finalize_idx=(finalize_idx if finalize_idx is not None else trigger_idx),
        probe_finalize_condition="no_retrace",
        parent_bos_anchor_idx=None,
        probe_input_idx=None,
        starting_idx=sub.starting_idx, direction=sub.direction, sub_tf=_SUB_TF,
        relative_dir=relative_dir, parent_floor_idx=parent_floor_idx,
        start_idx=start_idx,
        source_trigger=None,
        trigger_end_idx=trigger_end_idx,
        end_idx=(max(trigger_end_idx, start_idx) if trigger_end_idx is not None else None),
        end_reason=end_reason, ended_by_sub_id=ended_by,
        seq=seq,
    )
    registrar = getattr(pool, "add_record", None) if pool is not None else None
    if registrar is not None:
        registrar(rec)
    else:
        sub.records.append(rec)
    return rec


@contextlib.contextmanager
def _patched_probe(*, return_value=None, side_effect=None):
    """Patch `unified_probe` wherever the resolvers may bind it (the module
    attribute for a local import; `edm.unified_probe` for a module-level
    import), sharing ONE mock so call counts are exact."""
    mock = MagicMock(return_value=return_value, side_effect=side_effect)
    with ExitStack() as stack:
        stack.enter_context(patch.object(up_mod, "unified_probe", new=mock))
        if hasattr(edm, "unified_probe"):
            stack.enter_context(patch.object(edm, "unified_probe", new=mock))
        yield mock


@contextlib.contextmanager
def _patched_primitive(fake):
    """Patch `build_reference_zone_from_cts_event` (both possible bindings)."""
    mock = MagicMock(side_effect=fake)
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(rz_mod, "build_reference_zone_from_cts_event", new=mock),
        )
        if hasattr(edm, "build_reference_zone_from_cts_event"):
            stack.enter_context(
                patch.object(edm, "build_reference_zone_from_cts_event", new=mock),
            )
        yield mock


def _patched_fc_ref(zone):
    return patch.object(edm, "_build_first_confluence_ref_zone", return_value=zone)


def _patched_cts_derivation():
    """The sibling read passes `kl_zones=[]` (§5.1), so the primitive ALWAYS
    takes its ad-hoc branch; on the raw synthetic frame that derivation would
    return None. Pin the derived geometry so the tests exercise the SELECTION
    logic (which event wins, which idx is the input), not base-pattern
    derivation on synthetic data."""
    return patch.object(
        rz_mod, "_derive_cts_zone_ad_hoc", return_value=(0.6010, 0.6020, "sell"),
    )


def _probe_kwargs(mock_probe) -> dict:
    _, kwargs = mock_probe.call_args
    return kwargs


# ---------------------------------------------------------------------------
# _resolve_first_confluence_via_unified_probe — own ad-hoc BOS_0
# ---------------------------------------------------------------------------

class TestResolveFirstConfluence:
    """`pool=None` (the contract default) → no probe cache; every result here
    is a fresh probe run with `cache_hit=False`."""

    def _fixtures(self):
        h1 = _h1_df_uptrend(n_hours=10)
        m15 = _m15_from_h1(h1)
        return h1, m15

    def test_happy_path_returns_resolved_start_and_uses_probe_end_idx(self):
        h1, m15 = self._fixtures()
        # input: H1 3 price-mapped with -lower_sd = -1 → 4*3+1 = 13
        # end:   H1 7 price-mapped with +lower_sd = +1 → 4*7   = 28
        trig = _make_trigger(
            "first_confluence", parent_cycle_id=1,
            parent_sd=1, lower_sd=1,
            parent_input_idx=3, probe_end_idx=7,
        )
        fake = _fake_probe_result(
            starting_idx=15, finalize_idx=20, finalize_condition="no_retrace",
            bos0_inner=0.6020,
        )
        with _patched_fc_ref(_ref_zone("ad_hoc_bos_0", 13, inner=0.6020, outer=0.6000)) as mock_ref, \
                _patched_probe(return_value=fake) as mock_probe:
            res = _resolve_first_confluence_via_unified_probe(trig, h1, m15)

        bound = _FC_REF_SIG.bind(*mock_ref.call_args.args, **mock_ref.call_args.kwargs).arguments
        assert bound["m15_input_idx"] == 13
        assert bound["probe_direction"] == 1        # probe_direction = lower_sd

        kw = _probe_kwargs(mock_probe)
        assert kw["reference_zone"].source == "ad_hoc_bos_0"
        assert kw["direction"] == 1
        assert kw["input_idx"] == 13
        assert kw["enable_phase2"] is True          # first_confluence ONLY
        assert "end_idx" not in kw                  # §7 rename: the bound is `probe_end_idx`
        assert kw["probe_end_idx"] == 28

        assert isinstance(res, ResolvedStart)
        assert res.starting_idx == 15
        assert res.parent_bos_anchor_idx == 3        # the parent BOS anchor (H1) that seeded the probe
        assert res.bos0_inner == pytest.approx(0.6020)
        assert res.finalize_idx == 20               # FC keeps the probe's own finalize, raw
        assert res.finalize_condition == "no_retrace"
        assert res.probe_input_idx == 13
        assert res.cache_hit is False

    def test_missing_probe_meta_returns_probe_failure(self):
        h1, m15 = self._fixtures()
        trig = _make_trigger("first_confluence")
        trig.meta.pop("parent_input_idx")
        res = _resolve_first_confluence_via_unified_probe(trig, h1, m15)
        assert isinstance(res, ProbeFailure)
        assert res.probe_input_idx is None          # nothing was mapped yet
        assert "parent_input_idx" in res.detail     # names the H1 meta key (exported in the CSV `detail`)

    def test_input_mapping_failure_returns_probe_failure(self):
        # parent_df has 10 hours but the M15 frame only covers the first 3 →
        # H1 5's hour has no M15 candles → map_candle_to_lower_tf → None.
        h1 = _h1_df_uptrend(n_hours=10)
        m15 = _m15_from_h1(_h1_df_uptrend(n_hours=3))
        trig = _make_trigger(
            "first_confluence", parent_sd=1, lower_sd=1,
            parent_input_idx=5, probe_end_idx=7,
        )
        with _patched_probe(return_value=_fake_probe_result()) as mock_probe:
            res = _resolve_first_confluence_via_unified_probe(trig, h1, m15)
        mock_probe.assert_not_called()
        assert isinstance(res, ProbeFailure)
        # No M15 input exists on this branch: `ProbeFailure.probe_input_idx` is M15 or None,
        # never the H1 input 5 (PLAN_E §9.2 — the H1 value reaches an unresolved row as
        # `parent_input_idx`, from the sweep trigger).
        assert res.probe_input_idx is None

    def test_end_out_of_parent_bounds_returns_probe_failure_with_m15_input(self):
        h1, m15 = self._fixtures()
        # input H1 3 → 13 is mapped BEFORE the end is checked; end H1 99 is not in parent_df.
        trig = _make_trigger(
            "first_confluence", parent_sd=1, lower_sd=1,
            parent_input_idx=3, probe_end_idx=99,
        )
        with _patched_probe(return_value=_fake_probe_result()) as mock_probe:
            res = _resolve_first_confluence_via_unified_probe(trig, h1, m15)
        mock_probe.assert_not_called()
        assert isinstance(res, ProbeFailure)
        assert "probe_end_idx 99 out of parent bounds" in res.detail   # the END branch, not the input's
        assert res.probe_input_idx == 13            # the mapped M15 input, not the H1 3 (PLAN_E §9.2)

    def test_input_out_of_parent_bounds_returns_probe_failure_without_input(self):
        h1, m15 = self._fixtures()
        # H1 input 99 is not in parent_df: nothing is mapped → no M15 input (PLAN_E §9.2).
        trig = _make_trigger(
            "first_confluence", parent_sd=1, lower_sd=1,
            parent_input_idx=99, probe_end_idx=7,
        )
        with _patched_probe(return_value=_fake_probe_result()) as mock_probe:
            res = _resolve_first_confluence_via_unified_probe(trig, h1, m15)
        mock_probe.assert_not_called()
        assert isinstance(res, ProbeFailure)
        assert "parent_input_idx 99 out of parent bounds" in res.detail
        assert res.probe_input_idx is None

    def test_end_mapping_failure_returns_probe_failure_with_m15_input(self):
        # parent_df has 10 hours, the M15 frame covers the first 5: input H1 3 → 13 maps,
        # end H1 7's hour has no M15 candles → the end mapping fails.
        h1 = _h1_df_uptrend(n_hours=10)
        m15 = _m15_from_h1(_h1_df_uptrend(n_hours=5))
        trig = _make_trigger(
            "first_confluence", parent_sd=1, lower_sd=1,
            parent_input_idx=3, probe_end_idx=7,
        )
        with _patched_probe(return_value=_fake_probe_result()) as mock_probe:
            res = _resolve_first_confluence_via_unified_probe(trig, h1, m15)
        mock_probe.assert_not_called()
        assert isinstance(res, ProbeFailure)
        assert "end mapping failed" in res.detail
        assert res.probe_input_idx == 13            # the mapped M15 input, not the H1 3 (PLAN_E §9.2)

    def test_degenerate_window_returns_probe_failure_with_m15_input(self):
        h1, m15 = self._fixtures()
        # input H1 3 → 13 ; end H1 3 price-mapped +1 → 12 ; 12 <= 13 → degenerate
        trig = _make_trigger(
            "first_confluence", parent_sd=1, lower_sd=1,
            parent_input_idx=3, probe_end_idx=3,
        )
        with _patched_probe(return_value=_fake_probe_result()) as mock_probe:
            res = _resolve_first_confluence_via_unified_probe(trig, h1, m15)
        mock_probe.assert_not_called()
        assert isinstance(res, ProbeFailure)
        assert res.probe_input_idx == 13

    def test_ref_zone_none_returns_probe_failure(self):
        h1, m15 = self._fixtures()
        trig = _make_trigger(
            "first_confluence", parent_sd=1, lower_sd=1,
            parent_input_idx=3, probe_end_idx=7,
        )
        with _patched_fc_ref(None), \
                _patched_probe(return_value=_fake_probe_result()) as mock_probe:
            res = _resolve_first_confluence_via_unified_probe(trig, h1, m15)
        mock_probe.assert_not_called()
        assert isinstance(res, ProbeFailure)
        assert res.probe_input_idx == 13            # M15 input known by then

    def test_pending_probe_returns_probe_failure(self):
        h1, m15 = self._fixtures()
        trig = _make_trigger(
            "first_confluence", parent_sd=1, lower_sd=1,
            parent_input_idx=3, probe_end_idx=7,
        )
        pending = _fake_probe_result(
            starting_idx=0, status="pending",
            finalize_condition="no_cts_pending", finalize_idx=None,
        )
        with _patched_fc_ref(_ref_zone("ad_hoc_bos_0", 13, outer=0.6000)), \
                _patched_probe(return_value=pending):
            res = _resolve_first_confluence_via_unified_probe(trig, h1, m15)
        assert isinstance(res, ProbeFailure)
        assert res.probe_input_idx == 13


# ---------------------------------------------------------------------------
# _resolve_sibling_cts_via_unified_probe — sibling read is a POOL QUERY (§5.1)
# ---------------------------------------------------------------------------

class TestResolveSiblingCts:
    """Sibling reads under Plan C: `_build_sibling_cts_ref_zone_from_pool(pool,
    other_lens, S, C, probe_direction, idx_window, m15_df)`.

    Candidate records = `pool.records_for(other_lens, S, C)` with
    `direction == -probe_direction`, non-zero-length, `start_idx <= hi`, and
    `trigger_end_idx is None or >= lo`. Each record's events come from its
    sub's SLICE-LOCAL geometry shifted by `slice_begin`, clipped to
    `[max(lo, start_idx), min(hi, trigger_end_idx or hi)]`.
    """

    def _fixtures(self):
        h1 = _h1_df_uptrend(n_hours=10)
        m15 = _m15_from_h1(h1)
        return h1, m15

    # --- (1) first_counter reads the confluence lens -------------------------
    def test_first_counter_reads_confluence_lens_record_anchor(self):
        h1, m15 = self._fixtures()
        pool = SubStructurePool()
        # Confluence-lens sibling: +1 sub (= -probe_direction for a -1 probe),
        # slice_begin 2. Slice-local CTS_CONFIRMED idx 6 / anchor 4 →
        # entity-absolute idx 8 / extreme 6.
        sib = _stub_sub(pool, direction=1, starting_idx=4, slice_begin=2,
                        events=[_cts_event("CTS_CONFIRMED", 6, 4)])
        _rec(sib, pool=pool, lens="confluence", S=0, C=2, tss=0, trigger_type="first_confluence",
             trigger_idx=3, start_idx=4, seq=0)
        # Decoy on the COUNTER lens with a LATER CTS (entity 14): must be ignored
        # because first_counter reads the confluence lens.
        decoy = _stub_sub(pool, direction=1, starting_idx=10, slice_begin=0,
                          events=[_cts_event("CTS_UPDATED", 14)])
        _rec(decoy, pool=pool, lens="counter", S=0, C=2, tss=0, trigger_type="subsequent_counter",
             trigger_idx=10, start_idx=10, seq=1)

        trig = _make_trigger(
            "first_counter", parent_cycle_id=2, parent_sd=1, lower_sd=-1,
            probe_end_idx=5,       # trigger_event_idx 5 → hi = LOH(5) = 23
        )
        hi = 23
        fake = _fake_probe_result(starting_idx=12, source="cts_confirmed",
                                  finalize_idx=hi)   # Phase 1: finalize == hi
        with _patched_cts_derivation(), _patched_probe(return_value=fake) as mock_probe:
            res = _resolve_sibling_cts_via_unified_probe(
                trig, h1, m15, pool=pool, hi=hi,
            )
        kw = _probe_kwargs(mock_probe)
        assert kw["direction"] == -1
        assert kw["enable_phase2"] is False          # sibling path = Phase 1 only
        assert kw["input_idx"] == 6                  # winning CTS extreme, entity-absolute
        assert kw["reference_zone"].source == "cts_confirmed"
        assert kw["reference_zone"].anchor_idx == 6
        assert kw["probe_end_idx"] == hi
        assert isinstance(res, ResolvedStart)
        assert res.starting_idx == 12
        assert res.probe_input_idx == 6
        assert res.finalize_idx == hi
        assert res.cache_hit is False
        # PLAN_E Q4 (E5·4): FC-only — a sibling type has no parent BOS anchor; the
        # sibling CTS anchor it probes from is `probe_input_idx` (asserted above).
        assert res.parent_bos_anchor_idx is None

    # --- (2) subsequent_confluence reads the counter lens -------------------
    def test_subsequent_confluence_reads_counter_lens(self):
        h1, m15 = self._fixtures()
        pool = SubStructurePool()
        # Counter-lens sibling: -1 sub (= -probe_direction for a +1 probe).
        sib = _stub_sub(pool, direction=-1, starting_idx=4, slice_begin=0,
                        events=[_cts_event("CTS_CONFIRMED", 8, 6, struct_direction=-1)])
        _rec(sib, pool=pool, lens="counter", S=0, C=0, tss=0, trigger_type="first_counter",
             trigger_idx=4, start_idx=4, relative_dir="counter", seq=0)
        # Decoy on the CONFLUENCE lens (wrong lens), -1 with a LATER CTS at 14.
        decoy = _stub_sub(pool, direction=-1, starting_idx=10, slice_begin=0,
                          events=[_cts_event("CTS_UPDATED", 14, struct_direction=-1)])
        _rec(decoy, pool=pool, lens="confluence", S=0, C=0, tss=0, trigger_type="first_confluence",
             trigger_idx=10, start_idx=10, relative_dir="counter", seq=1)

        # prior_sd_trigger_idx H1 1 → lo = LOH(1) = 7 ; hi = 23. The sibling's
        # CTS_CONFIRMED idx 8 is inside [7, 23] (filter is on ev.idx); its
        # extreme (anchor 6) is the input.
        trig = _make_trigger(
            "subsequent_confluence", parent_cycle_id=0, parent_sd=1, lower_sd=1,
            probe_end_idx=5, prior_sd_trigger_idx=1,
        )
        hi = 23
        fake = _fake_probe_result(starting_idx=12, source="cts_confirmed", finalize_idx=hi)
        with _patched_cts_derivation(), _patched_probe(return_value=fake) as mock_probe:
            res = _resolve_sibling_cts_via_unified_probe(
                trig, h1, m15, pool=pool, hi=hi,
            )
        kw = _probe_kwargs(mock_probe)
        assert kw["direction"] == 1
        assert kw["input_idx"] == 6
        assert kw["probe_end_idx"] == hi
        assert isinstance(res, ResolvedStart)
        assert res.starting_idx == 12
        assert res.probe_input_idx == 6

    # --- (3) §5.1 concrete case: clip to the record's live window ----------
    def test_replaced_record_events_clipped_to_live_window(self, big_frames):
        """`subsequent_counter`(1,2)@4083 reads confluence (1,2). Sub A
        (-1, 3304) was replaced at 3819 but its geometry continues past it
        (CTS_UPDATED@3900); sub B (-1, 3760) is live from 3819 with a CTS@3850.
        The read must pick B's 3850, NOT A's 3900 (which would move
        `starting_idx` 4027 — a pool key)."""
        h1, m15 = big_frames
        pool = SubStructurePool()
        # Sub A: starting 3304 → slice_begin = 3304 - 50 = 3254.
        #   CTS_CONFIRMED entity 3750 (anchor 3740) → local 496 (anchor 486)
        #   CTS_UPDATED   entity 3900              → local 646
        sub_a = _stub_sub(pool, direction=-1, starting_idx=3304, slice_begin=3254,
                          events=[
                              _cts_event("CTS_CONFIRMED", 3750 - 3254, 3740 - 3254,
                                         struct_direction=-1),
                              _cts_event("CTS_UPDATED", 3900 - 3254, cycle_id=1,
                                         struct_direction=-1),
                          ])
        # Sub B: starting 3760 → slice_begin = 3710. CTS_UPDATED entity 3850 → local 140.
        sub_b = _stub_sub(pool, direction=-1, starting_idx=3760, slice_begin=3710,
                          events=[_cts_event("CTS_UPDATED", 3850 - 3710, struct_direction=-1)])
        # Records on the confluence lens, (1,2). A = FC (1,2) trigger 3611 →
        # finalize 3621 → start 3621, replaced at 3819 by B (same_dir_replacement).
        _rec(sub_a, pool=pool, lens="confluence", S=1, C=2, tss=0, trigger_type="first_confluence",
             trigger_idx=3611, finalize_idx=3621, start_idx=3621,
             trigger_end_idx=3819, end_reason="same_dir_replacement",
             ended_by=sub_b.sub_id, parent_floor_idx=3611, seq=0)
        # B = subsequent_confluence (1,2) trigger 3819 → start 3819, open.
        _rec(sub_b, pool=pool, lens="confluence", S=1, C=2, tss=1, trigger_type="subsequent_confluence",
             trigger_idx=3819, start_idx=3819, parent_floor_idx=3611, seq=1)

        # subsequent_counter (1,2): parent sid 1 is -1, probe direction +1,
        # prior_cts_prox_idx H1 926 → lo = LOH(926) = 3707 ;
        # trigger_event_idx H1 1020 → hi = LOH(1020) = 4083.
        #   A's clip: [max(3707, 3621), min(4083, 3819)] = [3707, 3819] → 3750 in, 3900 OUT
        #   B's clip: [max(3707, 3819), 4083]            = [3819, 4083] → 3850 in
        #   winner = max idx = 3850 (CTS_UPDATED → extreme = its idx)
        trig = _make_trigger(
            "subsequent_counter", parent_sid=1, parent_cycle_id=2,
            parent_sd=-1, lower_sd=1,
            probe_end_idx=1020, prior_cts_prox_idx=926,
        )
        hi = 4083
        fake = _fake_probe_result(starting_idx=4027, source="cts_updated", finalize_idx=hi)
        with _patched_cts_derivation(), _patched_probe(return_value=fake) as mock_probe:
            res = _resolve_sibling_cts_via_unified_probe(
                trig, h1, m15, pool=pool, hi=hi,
            )
        kw = _probe_kwargs(mock_probe)
        assert kw["input_idx"] == 3850
        assert kw["reference_zone"].anchor_idx == 3850
        assert kw["direction"] == 1
        assert kw["probe_end_idx"] == 4083
        assert isinstance(res, ResolvedStart)
        assert res.starting_idx == 4027
        assert res.probe_input_idx == 3850

    # --- (4) exists but not started by hi → excluded ------------------------
    def test_record_not_started_by_hi_excluded(self):
        h1, m15 = self._fixtures()
        pool = SubStructurePool()
        # An FC-like record: created at trigger 3, but start_idx 30 > hi 23.
        # Its geometry has a CTS inside [0, 23] (entity 8) — still excluded.
        sib = _stub_sub(pool, direction=1, starting_idx=4, slice_begin=0,
                        events=[_cts_event("CTS_CONFIRMED", 8, 6)])
        _rec(sib, pool=pool, lens="confluence", S=0, C=2, tss=0, trigger_type="first_confluence",
             trigger_idx=3, finalize_idx=30, start_idx=30)
        with _patched_cts_derivation():
            ref = _build_sibling_cts_ref_zone_from_pool(
                pool, "confluence", 0, 2, -1, (0, 23), m15,
            )
        assert ref is None

    # --- (5) zero-length records excluded -----------------------------------
    def test_zero_length_record_excluded(self):
        h1, m15 = self._fixtures()
        pool = SubStructurePool()
        sib = _stub_sub(pool, direction=1, starting_idx=4, slice_begin=0,
                        events=[_cts_event("CTS_CONFIRMED", 8, 6)])
        # trigger_end_idx 8 <= start_idx 8 → zero-length (defined on
        # trigger_end_idx, §2.1). Its CTS@8 would otherwise sit inside its own
        # [8, 8] clip — exclusion must be by the zero-length rule.
        rec = _rec(sib, pool=pool, lens="confluence", S=0, C=2, tss=0, trigger_type="first_confluence",
                   trigger_idx=8, start_idx=8, trigger_end_idx=8, end_reason="parent_end")
        assert rec.is_zero_length
        with _patched_cts_derivation():
            ref = _build_sibling_cts_ref_zone_from_pool(
                pool, "confluence", 0, 2, -1, (0, 23), m15,
            )
        assert ref is None

    # --- (6) ended before lo → excluded -------------------------------------
    def test_record_ended_before_lo_excluded(self):
        h1, m15 = self._fixtures()
        pool = SubStructurePool()
        # Record live [4, 9]; window lo = 11 (subsequent_counter with
        # prior_cts_prox_idx H1 2 → LOH(2) = 11). trigger_end_idx 9 < lo 11 →
        # not live in the window. Its geometry runs on to the data edge and has
        # a CTS_UPDATED@15 inside [11, 23] — it must NOT be a candidate.
        sib = _stub_sub(pool, direction=1, starting_idx=4, slice_begin=0,
                        events=[_cts_event("CTS_CONFIRMED", 8, 6),
                                _cts_event("CTS_UPDATED", 15)])
        _rec(sib, pool=pool, lens="confluence", S=0, C=2, tss=0, trigger_type="first_confluence",
             trigger_idx=4, start_idx=4, trigger_end_idx=9, end_reason="reversal")
        with _patched_cts_derivation():
            ref = _build_sibling_cts_ref_zone_from_pool(
                pool, "confluence", 0, 2, -1, (11, 23), m15,
            )
        assert ref is None

    # --- (7) no candidates → None → caller's ad-hoc fallback ----------------
    def test_no_candidates_returns_none_and_caller_falls_back(self):
        h1, m15 = self._fixtures()
        pool = SubStructurePool()           # no records at all
        trig = _make_trigger("first_counter", parent_cycle_id=2, lower_sd=-1, probe_end_idx=5)
        hi = 23
        # Pool query alone → None.
        assert _build_sibling_cts_ref_zone_from_pool(
            pool, "confluence", 0, 2, -1, (0, hi), m15,
        ) is None
        # Resolver: fallback = own-frame window extreme on the -lower_sd = +1
        # side over [0, 23] → highest high = idx 20 (verified on the fixture),
        # then the ad-hoc BOS_0 at that candle (patched) with anchor_idx 6.
        fb_ref = _ref_zone("ad_hoc_bos_0", 6, inner=0.6020, outer=0.6010, side="sell")
        fake = _fake_probe_result(starting_idx=9, source="ad_hoc_bos_0", finalize_idx=hi)
        with _patched_fc_ref(fb_ref) as mock_fb, _patched_probe(return_value=fake) as mock_probe:
            res = _resolve_sibling_cts_via_unified_probe(
                trig, h1, m15, pool=pool, hi=hi,
            )
        mock_fb.assert_called_once()
        bound = _FC_REF_SIG.bind(*mock_fb.call_args.args, **mock_fb.call_args.kwargs).arguments
        assert bound["m15_input_idx"] == 20
        assert bound["probe_direction"] == -1
        kw = _probe_kwargs(mock_probe)
        assert kw["reference_zone"].source == "ad_hoc_bos_0"
        assert kw["input_idx"] == 6                  # fallback ref's anchor_idx
        assert kw["probe_end_idx"] == hi
        assert isinstance(res, ResolvedStart)
        assert res.starting_idx == 9
        assert res.probe_input_idx == 6
        assert res.parent_bos_anchor_idx is None      # FC-only (PLAN_E Q4), see (1)

    # --- (8) both unavailable → ProbeFailure --------------------------------
    def test_both_unavailable_returns_probe_failure(self):
        h1, m15 = self._fixtures()
        pool = SubStructurePool()
        trig = _make_trigger("first_counter", parent_cycle_id=2, lower_sd=-1, probe_end_idx=5)
        with _patched_fc_ref(None), _patched_probe(return_value=_fake_probe_result()) as mock_probe:
            res = _resolve_sibling_cts_via_unified_probe(
                trig, h1, m15, pool=pool, hi=23,
            )
        mock_probe.assert_not_called()
        assert isinstance(res, ProbeFailure)
        assert res.probe_input_idx is None

    # --- (9) degenerate: input >= hi → ProbeFailure carrying the input ------
    def test_degenerate_input_at_hi_returns_probe_failure_with_input(self):
        h1, m15 = self._fixtures()
        pool = SubStructurePool()
        # CTS_CONFIRMED at entity 23 (== hi, inside the window) with its anchor
        # AT 23 → ref builds with anchor_idx 23, but 23 >= hi 23 → no
        # forward scan window.
        sib = _stub_sub(pool, direction=1, starting_idx=4, slice_begin=0,
                        events=[_cts_event("CTS_CONFIRMED", 23, 23)])
        _rec(sib, pool=pool, lens="confluence", S=0, C=2, tss=0, trigger_type="first_confluence",
             trigger_idx=4, start_idx=4)
        trig = _make_trigger("first_counter", parent_cycle_id=2, lower_sd=-1, probe_end_idx=5)
        with _patched_cts_derivation(), \
                _patched_probe(return_value=_fake_probe_result()) as mock_probe:
            res = _resolve_sibling_cts_via_unified_probe(
                trig, h1, m15, pool=pool, hi=23,
            )
        mock_probe.assert_not_called()
        assert isinstance(res, ProbeFailure)
        assert res.probe_input_idx == 23             # reported for traceability

    # --- (9b) pending probe → ProbeFailure carrying the M15 input -----------
    def test_pending_probe_returns_probe_failure_with_m15_input(self):
        h1, m15 = self._fixtures()
        pool = SubStructurePool()
        # The confluence-lens sibling of (1): slice-local CTS_CONFIRMED 6 / anchor 4,
        # slice_begin 2 → co-sourced M15 input 6. The probe does not finalize →
        # ProbeFailure carrying that M15 input (the unresolved row's `probe_input_idx`,
        # PLAN_E §9.2) — never None, never the trigger's H1 input 1.
        sib = _stub_sub(pool, direction=1, starting_idx=4, slice_begin=2,
                        events=[_cts_event("CTS_CONFIRMED", 6, 4)])
        _rec(sib, pool=pool, lens="confluence", S=0, C=2, tss=0, trigger_type="first_confluence",
             trigger_idx=3, start_idx=4, seq=0)
        trig = _make_trigger("first_counter", parent_cycle_id=2, lower_sd=-1, probe_end_idx=5)
        pending = _fake_probe_result(
            starting_idx=0, status="pending",
            finalize_condition="no_cts_pending", finalize_idx=None,
        )
        with _patched_cts_derivation(), _patched_probe(return_value=pending) as mock_probe:
            res = _resolve_sibling_cts_via_unified_probe(
                trig, h1, m15, pool=pool, hi=23,
            )
        mock_probe.assert_called_once()
        assert isinstance(res, ProbeFailure)
        assert res.detail == "probe pending"
        assert res.probe_input_idx == 6

    # --- (10) primitive gets kl_zones=[] + the SHARED frame + shifted copies -
    def test_primitive_called_with_empty_kl_zones_and_shared_frame(self):
        h1, m15 = self._fixtures()
        pool = SubStructurePool()
        # slice_begin 2: local CTS_CONFIRMED idx 6 / anchor 4 / confirmed_at 6
        # → entity-absolute 8 / 6 / 8.
        local_ev = _cts_event("CTS_CONFIRMED", 6, 4)
        sib = _stub_sub(pool, direction=1, starting_idx=4, slice_begin=2, events=[local_ev])
        _rec(sib, pool=pool, lens="confluence", S=0, C=2, tss=0, trigger_type="first_confluence",
             trigger_idx=3, start_idx=4)

        def _fake_primitive(*args, **kwargs):
            return _ref_zone("cts_confirmed", 6, inner=0.6020, outer=0.6010, side="sell")

        with _patched_primitive(_fake_primitive) as mock_prim:
            ref = _build_sibling_cts_ref_zone_from_pool(
                pool, "confluence", 0, 2, -1, (0, 23), m15,
            )
        assert ref is not None and ref.anchor_idx == 6
        mock_prim.assert_called_once()
        bound = _PRIMITIVE_SIG.bind(*mock_prim.call_args.args, **mock_prim.call_args.kwargs).arguments
        assert bound["kl_zones"] == []               # §5.1: the CONFIRMED-zone branch is dead for subs
        assert bound["df"] is m15                    # the shared entity-absolute frame, NOT bounded.df
        assert bound["sid"] == 0
        assert bound["probe_direction"] == -1
        assert bound["idx_window"] == (0, 23)
        passed = bound["events"]
        assert len(passed) == 1
        ev = passed[0]
        assert ev.type == "CTS_CONFIRMED"
        assert int(ev.idx) == 8                      # shifted by slice_begin
        assert int(ev.meta["cts_anchor_idx"]) == 6   # idx-bearing meta shifted too
        assert int(ev.meta["confirmed_at"]) == 8
        # Entity-absolute COPIES: the shared geometry object is untouched.
        assert ev is not local_ev
        assert int(local_ev.idx) == 6
        assert int(local_ev.meta["cts_anchor_idx"]) == 4

    # --- (11) direction filter is on the RECORD's direction -----------------
    def test_direction_filter_is_on_record_direction(self):
        h1, m15 = self._fixtures()
        pool = SubStructurePool()
        # Both records on the confluence lens, (0,2), both live over [0, 23]
        # (opposite-direction overlap is allowed, §17.5).
        plus = _stub_sub(pool, direction=1, starting_idx=4, slice_begin=0,
                         events=[_cts_event("CTS_CONFIRMED", 8, 6, struct_direction=1)])
        _rec(plus, pool=pool, lens="confluence", S=0, C=2, tss=0, trigger_type="first_confluence",
             trigger_idx=4, start_idx=4, seq=0)
        minus = _stub_sub(pool, direction=-1, starting_idx=10, slice_begin=0,
                          events=[_cts_event("CTS_UPDATED", 14, struct_direction=-1)])
        _rec(minus, pool=pool, lens="confluence", S=0, C=2, tss=1, trigger_type="reversal",
             trigger_idx=10, start_idx=10, relative_dir="counter", seq=1)

        with _patched_cts_derivation():
            # probe -1 → only the +1 record qualifies → its extreme 6, NOT the
            # later -1 CTS at 14.
            ref_minus_probe = _build_sibling_cts_ref_zone_from_pool(
                pool, "confluence", 0, 2, -1, (0, 23), m15,
            )
            # probe +1 → only the -1 record qualifies → 14.
            ref_plus_probe = _build_sibling_cts_ref_zone_from_pool(
                pool, "confluence", 0, 2, 1, (0, 23), m15,
            )
        assert ref_minus_probe is not None and ref_minus_probe.anchor_idx == 6
        assert ref_plus_probe is not None and ref_plus_probe.anchor_idx == 14

    # --- (12) the probe bound is the PASSED hi, not the trigger's meta ------
    def test_probe_end_is_the_passed_hi(self):
        h1, m15 = self._fixtures()
        pool = SubStructurePool()
        sib = _stub_sub(pool, direction=1, starting_idx=4, slice_begin=0,
                        events=[_cts_event("CTS_CONFIRMED", 8, 6)])
        _rec(sib, pool=pool, lens="confluence", S=0, C=2, tss=0, trigger_type="first_confluence",
             trigger_idx=4, start_idx=4)
        # meta probe_end_idx 5 would map to 23; the sweep passes hi = 27.
        trig = _make_trigger("first_counter", parent_cycle_id=2, lower_sd=-1, probe_end_idx=5)
        hi = 27
        fake = _fake_probe_result(starting_idx=12, source="cts_confirmed", finalize_idx=hi)
        with _patched_cts_derivation(), _patched_probe(return_value=fake) as mock_probe:
            res = _resolve_sibling_cts_via_unified_probe(
                trig, h1, m15, pool=pool, hi=hi,
            )
        assert _probe_kwargs(mock_probe)["probe_end_idx"] == 27
        assert isinstance(res, ResolvedStart) and res.finalize_idx == 27


# ---------------------------------------------------------------------------
# _sibling_cts_idx_window — per-variation lower bound (unchanged semantics;
# `hi` is now the reading trigger's trigger_idx, passed in by the sweep).
# These four pin behaviour the contract says SURVIVES Plan C.
# ---------------------------------------------------------------------------

class TestSiblingCtsIdxWindow:

    def test_first_counter_lo_is_zero(self):
        h1 = _h1_df_uptrend(n_hours=5)
        m15 = _m15_from_h1(h1)
        trig = _make_trigger("first_counter", probe_end_idx=4)
        lo, hi = _sibling_cts_idx_window(trig, h1, m15, 19)
        assert lo == 0
        assert hi == 19

    def test_subsequent_confluence_lo_from_prior_sd(self):
        h1 = _h1_df_uptrend(n_hours=5)
        m15 = _m15_from_h1(h1)
        trig = _make_trigger(
            "subsequent_confluence", lower_sd=1,
            probe_end_idx=4, prior_sd_trigger_idx=1,
        )
        lo, hi = _sibling_cts_idx_window(trig, h1, m15, 19)
        # prior_sd H1 idx 1 → last-M15-of-hour = 4*1+3 = 7
        assert lo == 7
        assert hi == 19

    def test_subsequent_counter_lo_from_prior_cts(self):
        h1 = _h1_df_uptrend(n_hours=5)
        m15 = _m15_from_h1(h1)
        trig = _make_trigger(
            "subsequent_counter", lower_sd=-1,
            probe_end_idx=4, prior_cts_prox_idx=2,
        )
        lo, hi = _sibling_cts_idx_window(trig, h1, m15, 19)
        # prior_cts H1 idx 2 → last-M15-of-hour = 4*2+3 = 11
        assert lo == 11
        assert hi == 19

    def test_missing_prior_meta_degrades_to_zero(self):
        h1 = _h1_df_uptrend(n_hours=5)
        m15 = _m15_from_h1(h1)
        trig = _make_trigger("subsequent_confluence", lower_sd=1, probe_end_idx=4)
        lo, hi = _sibling_cts_idx_window(trig, h1, m15, 19)
        assert lo == 0
        assert hi == 19


# ---------------------------------------------------------------------------
# _resolve_trigger_m15_start dispatcher — routing + pool/hi threading
# ---------------------------------------------------------------------------

class TestDispatcher:

    def test_first_confluence_routes_to_fc_resolver_and_forwards_pool(self):
        h1 = _h1_df_uptrend(n_hours=5)
        m15 = _m15_from_h1(h1)
        pool = SubStructurePool()
        trig = _make_trigger(
            "first_confluence", parent_sd=1, lower_sd=1,
            parent_input_idx=2, probe_end_idx=4,
        )
        fake = ResolvedStart(
            starting_idx=77, parent_bos_anchor_idx=2, bos0_inner=0.6020,
            finalize_idx=80, finalize_condition="no_retrace",
        )
        with patch.object(edm, "_resolve_first_confluence_via_unified_probe",
                          return_value=fake) as mock_conf, \
                patch.object(edm, "_resolve_sibling_cts_via_unified_probe") as mock_sib:
            out = _resolve_trigger_m15_start(trig, h1, m15, pool=pool, hi=19)
        mock_conf.assert_called_once()
        mock_sib.assert_not_called()
        assert out is fake
        _, kwargs = mock_conf.call_args
        assert kwargs["pool"] is pool

    @pytest.mark.parametrize(
        "use_case,lower_sd",
        [
            ("first_counter", -1),
            ("subsequent_confluence", 1),
            ("subsequent_counter", -1),
        ],
    )
    def test_sibling_variations_route_to_sibling_resolver_with_pool_and_hi(
        self, use_case, lower_sd,
    ):
        h1 = _h1_df_uptrend(n_hours=5)
        m15 = _m15_from_h1(h1)
        pool = SubStructurePool()
        trig = _make_trigger(use_case, lower_sd=lower_sd, probe_end_idx=3)
        fake = ResolvedStart(
            starting_idx=99, parent_bos_anchor_idx=6, bos0_inner=0.6020,
            finalize_idx=15, finalize_condition="no_retrace",
        )
        with patch.object(edm, "_resolve_sibling_cts_via_unified_probe",
                          return_value=fake) as mock_sib, \
                patch.object(edm, "_resolve_first_confluence_via_unified_probe") as mock_conf:
            out = _resolve_trigger_m15_start(trig, h1, m15, pool=pool, hi=15)
        mock_conf.assert_not_called()
        mock_sib.assert_called_once()
        _, kwargs = mock_sib.call_args
        assert kwargs["pool"] is pool
        assert kwargs["hi"] == 15
        assert out is fake

    def test_legacy_escape_hatch_is_gone(self):
        # §7: `_LEGACY_PROBE_USE_CASES` / `_resolve_via_legacy_probe` deleted,
        # and `lower_tf_pipeline.py` deleted outright.
        assert not hasattr(edm, "_LEGACY_PROBE_USE_CASES")
        assert not hasattr(edm, "_resolve_via_legacy_probe")
        with pytest.raises(ImportError):
            importlib.import_module("engine_v2.multitf.lower_tf_pipeline")

    def test_unknown_use_case_raises(self):
        h1 = _h1_df_uptrend(n_hours=5)
        m15 = _m15_from_h1(h1)
        trig = _make_trigger("not_a_use_case", probe_end_idx=3)
        # PLAN-AMBIGUITY: the plan says the legacy default is retired and an
        # unknown use_case must raise, but does not name the exception type.
        with pytest.raises((ValueError, KeyError, AssertionError)):
            _resolve_trigger_m15_start(trig, h1, m15, pool=SubStructurePool(), hi=15)

    def test_sibling_entity_df_kwarg_is_gone(self):
        h1 = _h1_df_uptrend(n_hours=5)
        m15 = _m15_from_h1(h1)
        trig = _make_trigger("first_counter", probe_end_idx=3)
        with pytest.raises(TypeError):
            _resolve_trigger_m15_start(  # type: ignore[call-arg]
                trig, h1, m15, sibling_entity_df=m15.copy(),
            )

    def test_resolved_types_are_the_sweep_definitions(self):
        # Contract: ResolvedStart / ProbeFailure are defined ONCE in
        # lifecycle_sweep and re-exported by entity_df_mutation.
        ls = importlib.import_module("engine_v2.multitf.lifecycle_sweep")
        assert edm.ResolvedStart is ls.ResolvedStart
        assert edm.ProbeFailure is ls.ProbeFailure


# ---------------------------------------------------------------------------
# Probe cache (§5.3) — keyed (parent_path, "M15", direction, initial_input_idx)
# ---------------------------------------------------------------------------

class TestProbeCache:
    """Driven through the first_confluence resolver with a REAL pool. FC
    fixtures: input H1 3 → M15 13 (−lower_sd = −1 → 4*3+1); end H1 7 → 28
    (+1 → 4*7); a second FC input H1 4 → M15 17 (4*4+1); end H1 8 → 32."""

    def _fixtures(self):
        h1 = _h1_df_uptrend(n_hours=10)
        m15 = _m15_from_h1(h1)
        return h1, m15

    @staticmethod
    def _fc(parent_input_idx: int = 3, probe_end_idx: int = 7) -> MultiTFTrigger:
        return _make_trigger(
            "first_confluence", parent_cycle_id=1, parent_sd=1, lower_sd=1,
            parent_input_idx=parent_input_idx, probe_end_idx=probe_end_idx,
        )

    @staticmethod
    def _zone(inner: float = 0.6020) -> ReferenceZone:
        return _ref_zone("ad_hoc_bos_0", 13, inner=inner, outer=0.6000, side="buy")

    def test_miss_runs_probe_and_records_entry(self, capsys):
        h1, m15 = self._fixtures()
        pool = SubStructurePool()
        assert pool.get_cached_probe(_PARENT_PATH, _SUB_TF, 1, 13) is None
        fake = _fake_probe_result(starting_idx=15, finalize_idx=20,
                                  finalize_condition="no_retrace", bos0_inner=0.6020)
        with _patched_fc_ref(self._zone()), _patched_probe(return_value=fake) as mock_probe:
            res = _resolve_first_confluence_via_unified_probe(
                self._fc(), h1, m15, pool=pool,
            )
        assert mock_probe.call_count == 1
        assert isinstance(res, ResolvedStart) and res.cache_hit is False
        entry = pool.get_cached_probe(_PARENT_PATH, _SUB_TF, 1, 13)
        # `ref_inner` = the reference zone the probe was RUN against (iteration
        # 1's threshold) — the tripwire's comparand (cold review 2026-09-20: the
        # final-iteration bos0_inner moves on a reset, so it is NOT the comparand).
        assert entry == ProbeCacheEntry(
            starting_idx=15, finalize_idx=20, finalize_condition="no_retrace",
            bos0_inner=0.6020, probe_end_idx=28, ref_inner=0.6020,
        )
        assert "[probe_cache] miss" in capsys.readouterr().out

    def test_same_direction_and_input_hits_and_skips_probe(self):
        h1, m15 = self._fixtures()
        pool = SubStructurePool()
        fake = _fake_probe_result(starting_idx=15, finalize_idx=20,
                                  finalize_condition="no_retrace", bos0_inner=0.6020)
        with _patched_fc_ref(self._zone()), _patched_probe(return_value=fake) as mock_probe:
            r1 = _resolve_first_confluence_via_unified_probe(self._fc(), h1, m15, pool=pool)
            assert mock_probe.call_count == 1
            # Same (direction +1, input 13), same bound → hit; unified_probe NOT re-run.
            r2 = _resolve_first_confluence_via_unified_probe(self._fc(), h1, m15, pool=pool)
            assert mock_probe.call_count == 1
        assert r1.cache_hit is False
        assert isinstance(r2, ResolvedStart)
        assert r2.cache_hit is True
        assert r2.starting_idx == 15
        assert r2.finalize_idx == 20                # inherited raw (§2.1)
        assert r2.finalize_condition == "no_retrace"
        assert r2.bos0_inner == pytest.approx(0.6020)
        assert r2.probe_input_idx == 13

    def test_hit_logs_exact_or_approx_by_probe_end(self, capsys):
        h1, m15 = self._fixtures()
        pool = SubStructurePool()
        fake = _fake_probe_result(starting_idx=15, finalize_idx=20, bos0_inner=0.6020)
        with _patched_fc_ref(self._zone()), _patched_probe(return_value=fake) as mock_probe:
            _resolve_first_confluence_via_unified_probe(self._fc(), h1, m15, pool=pool)
            capsys.readouterr()
            # Same bound (end H1 7 → 28) → exact hit.
            _resolve_first_confluence_via_unified_probe(self._fc(), h1, m15, pool=pool)
            out_exact = capsys.readouterr().out
            # Different bound (end H1 8 → 32 ≠ cached 28) → APPROX hit.
            r3 = _resolve_first_confluence_via_unified_probe(
                self._fc(probe_end_idx=8), h1, m15, pool=pool,
            )
            out_approx = capsys.readouterr().out
            assert mock_probe.call_count == 1
        assert "[probe_cache] hit" in out_exact
        assert "APPROX" not in out_exact
        assert "[probe_cache] APPROX hit" in out_approx
        assert r3.cache_hit is True and r3.finalize_idx == 20

    def test_hit_logs_ref_zone_differs_when_inner_mismatch(self, capsys):
        h1, m15 = self._fixtures()
        pool = SubStructurePool()
        # Cached bos0_inner = iteration 1's BOS_0 threshold = the reference inner (0.6020).
        fake = _fake_probe_result(starting_idx=15, finalize_idx=20, bos0_inner=0.6020)
        with _patched_probe(return_value=fake):
            with _patched_fc_ref(self._zone(inner=0.6020)):
                _resolve_first_confluence_via_unified_probe(self._fc(), h1, m15, pool=pool)
                capsys.readouterr()
                # Same inner → hit without the tripwire.
                _resolve_first_confluence_via_unified_probe(self._fc(), h1, m15, pool=pool)
                out_same = capsys.readouterr().out
            with _patched_fc_ref(self._zone(inner=0.6050)):
                # Reference inner 0.6050 vs cached 0.6020 → not isclose → tripwire.
                r = _resolve_first_confluence_via_unified_probe(self._fc(), h1, m15, pool=pool)
                out_diff = capsys.readouterr().out
        assert "REF-ZONE DIFFERS" not in out_same
        assert "[probe_cache] REF-ZONE DIFFERS" in out_diff
        # Still a hit — first probe is truth (design, §5.3).
        assert r.cache_hit is True and r.bos0_inner == pytest.approx(0.6020)

    def test_tripwire_compares_reference_inner_not_final_bos0_inner(self, capsys):
        """Cold review 2026-09-20: the cached probe's `bos0_inner` is the FINAL
        iteration's threshold (it moves on every reset); the tripwire's comparand
        is the reference inner the probe was RUN against (`ProbeCacheEntry.ref_inner`).
        A probe that reset (final bos0_inner 0.6100 != reference 0.6020) must NOT
        trip when the hitting trigger's reference is the same 0.6020, and MUST trip
        when the hitting reference equals the final bos0_inner but not the reference."""
        h1, m15 = self._fixtures()
        pool = SubStructurePool()
        fake = _fake_probe_result(starting_idx=15, finalize_idx=20, bos0_inner=0.6100)
        with _patched_probe(return_value=fake):
            with _patched_fc_ref(self._zone(inner=0.6020)):
                _resolve_first_confluence_via_unified_probe(self._fc(), h1, m15, pool=pool)
                capsys.readouterr()
                r_same = _resolve_first_confluence_via_unified_probe(self._fc(), h1, m15, pool=pool)
                out_same = capsys.readouterr().out
            with _patched_fc_ref(self._zone(inner=0.6100)):
                r_diff = _resolve_first_confluence_via_unified_probe(self._fc(), h1, m15, pool=pool)
                out_diff = capsys.readouterr().out
        entry = pool.get_cached_probe(_PARENT_PATH, _SUB_TF, 1, 13)
        assert entry.ref_inner == pytest.approx(0.6020) and entry.bos0_inner == pytest.approx(0.6100)
        assert r_same.cache_hit and "REF-ZONE DIFFERS" not in out_same
        assert r_diff.cache_hit and "[probe_cache] REF-ZONE DIFFERS" in out_diff

    def test_different_input_converging_keeps_own_finalize(self):
        h1, m15 = self._fixtures()
        pool = SubStructurePool()
        finalize_by_input = {13: 20, 17: 24}

        def _probe(*args, **kwargs):
            inp = kwargs.get("input_idx", args[1] if len(args) > 1 else None)
            return _fake_probe_result(
                starting_idx=15, finalize_idx=finalize_by_input[int(inp)],
                bos0_inner=0.6020,
            )

        with _patched_probe(side_effect=_probe) as mock_probe:
            with _patched_fc_ref(_ref_zone("ad_hoc_bos_0", 13, inner=0.6020, outer=0.6000)):
                ra = _resolve_first_confluence_via_unified_probe(
                    self._fc(parent_input_idx=3), h1, m15, pool=pool,
                )
            with _patched_fc_ref(_ref_zone("ad_hoc_bos_0", 17, inner=0.6020, outer=0.6000)):
                rb = _resolve_first_confluence_via_unified_probe(
                    self._fc(parent_input_idx=4), h1, m15, pool=pool,
                )
            # Input 17 ≠ 13 → its OWN probe ran (the pool, not the cache, dedups the MS).
            assert mock_probe.call_count == 2
        assert ra.starting_idx == rb.starting_idx == 15
        assert ra.cache_hit is False and rb.cache_hit is False
        assert ra.finalize_idx == 20
        assert rb.finalize_idx == 24                # keeps its own finalize
        assert pool.get_cached_probe(_PARENT_PATH, _SUB_TF, 1, 13).finalize_idx == 20
        assert pool.get_cached_probe(_PARENT_PATH, _SUB_TF, 1, 17).finalize_idx == 24

    def test_real_probe_is_deterministic_on_identical_input_and_bound(self):
        # Real Phase-1 unified_probe on the prepared M15 fixture (verified:
        # finalized / no_retrace / starting_idx 0 / finalize_idx == probe_end_idx 30).
        from engine_v2.features.candle_classifier import apply_candle_classification
        from engine_v2.patterns.imbalance import compute_imbalance
        from engine_v2.patterns.pattern_engine import detect_patterns

        m15 = _m15_from_h1(_h1_df_uptrend(n_hours=10))
        df = compute_imbalance(detect_patterns(apply_candle_classification(m15).df).df)
        df.attrs["pair"] = "NZD_USD"
        ref = _ref_zone("ad_hoc_bos_0", 0, inner=0.6000, outer=0.5990, side="buy")
        r1 = up_mod.unified_probe(df, 0, 1, ref, probe_end_idx=30, timeframe="M15")
        r2 = up_mod.unified_probe(df, 0, 1, ref, probe_end_idx=30, timeframe="M15")
        assert r1 == r2
        assert r1.status == "finalized"
        assert (r1.starting_idx, r1.finalize_idx, r1.finalize_condition) == (
            r2.starting_idx, r2.finalize_idx, r2.finalize_condition,
        )
        assert r1.finalize_idx == 30                # Phase 1: finalize == probe_end_idx
        assert not hasattr(r1, "start_idx")         # §7 rename: ProbeResult.starting_idx

    def test_pool_cache_primitives_first_write_wins(self):
        pool = SubStructurePool()
        key = (_PARENT_PATH, _SUB_TF, -1, 6)
        assert pool.get_cached_probe(*key) is None
        e1 = ProbeCacheEntry(starting_idx=12, finalize_idx=23, finalize_condition="no_retrace",
                             bos0_inner=0.6020, probe_end_idx=23)
        pool.record_probe(*key, e1)
        pool.record_probe(*key, e1)                 # same entry: no-op
        assert pool.get_cached_probe(*key) == e1
        e2 = dataclasses.replace(e1, finalize_idx=27)
        with pytest.raises(AssertionError):
            pool.record_probe(*key, e2)             # different entry: first write wins, raise
        assert pool.get_cached_probe(*key) == e1


# ---------------------------------------------------------------------------
# §5.2 — sibling types: finalize_idx == hi when the probe RAN; cache hit exempt
# ---------------------------------------------------------------------------

class TestFinalizeEqualsTrigger:

    def _pool_with_confluence_record(self):
        pool = SubStructurePool()
        sib = _stub_sub(pool, direction=1, starting_idx=4, slice_begin=0,
                        events=[_cts_event("CTS_CONFIRMED", 8, 6)])
        _rec(sib, pool=pool, lens="confluence", S=0, C=2, tss=0, trigger_type="first_confluence",
             trigger_idx=4, start_idx=4)
        return pool

    def test_sibling_probe_run_with_finalize_not_hi_raises(self):
        h1 = _h1_df_uptrend(n_hours=10)
        m15 = _m15_from_h1(h1)
        pool = self._pool_with_confluence_record()
        trig = _make_trigger("first_counter", parent_cycle_id=2, lower_sd=-1, probe_end_idx=5)
        # A Phase-1 probe finalizes AT its bound by construction; a result whose
        # finalize_idx (20) != hi (23) violates §5.2 → the resolver asserts.
        fake = _fake_probe_result(starting_idx=12, source="cts_confirmed", finalize_idx=20)
        with _patched_cts_derivation(), _patched_probe(return_value=fake):
            with pytest.raises(AssertionError):
                _resolve_sibling_cts_via_unified_probe(trig, h1, m15, pool=pool, hi=23)

    def test_cache_hit_record_is_exempt(self):
        h1 = _h1_df_uptrend(n_hours=10)
        m15 = _m15_from_h1(h1)
        pool = self._pool_with_confluence_record()
        # First read at hi = 23 (trigger_event_idx 5): probe runs, finalize 23 == hi
        # → cached under (H1.main, M15, -1, input 6).
        t1 = _make_trigger("first_counter", parent_cycle_id=2, lower_sd=-1, probe_end_idx=5)
        # Second read at hi = 27 (trigger_event_idx 6 → LOH 27): same direction,
        # same sibling input 6 → cache HIT; the inherited finalize 23 != 27 is
        # allowed (the re-trigger does not wait — §2.1), no assert.
        t2 = _make_trigger("first_counter", parent_cycle_id=2, lower_sd=-1, probe_end_idx=6)
        fake = _fake_probe_result(starting_idx=12, source="cts_confirmed", finalize_idx=23)
        with _patched_cts_derivation(), _patched_probe(return_value=fake) as mock_probe:
            r1 = _resolve_sibling_cts_via_unified_probe(t1, h1, m15, pool=pool, hi=23)
            r2 = _resolve_sibling_cts_via_unified_probe(t2, h1, m15, pool=pool, hi=27)
            assert mock_probe.call_count == 1
        assert isinstance(r1, ResolvedStart) and r1.cache_hit is False and r1.finalize_idx == 23
        assert isinstance(r2, ResolvedStart)
        assert r2.cache_hit is True
        assert r2.finalize_idx == 23                # inherited raw, not 27
        assert r2.starting_idx == 12
        assert r2.probe_input_idx == 6
        cached = pool.get_cached_probe(_PARENT_PATH, _SUB_TF, -1, 6)
        assert cached is not None and cached.finalize_idx == 23 and cached.probe_end_idx == 23


class TestSiblingClipOnTheMoment:
    """Plan E E3b: the sibling read clips each record's CTS events on their
    MOMENT, not the stamped anchor — a CTS_ESTABLISHED anchored inside the
    window but established after it is not yet knowable to the reader."""

    def _pool(self):
        from engine_v2.tests._event_factory import make_cts_established
        pool = SubStructurePool()
        est = make_cts_established(cts_anchor_idx=20, confirmed_at=26, price=0.6020,
                                   structure_id=0, cycle_id=0, struct_direction=1)
        sib = _stub_sub(pool, direction=1, starting_idx=5, slice_begin=0, events=[est])
        _rec(sib, pool=pool, lens="confluence", S=0, C=2, tss=0, trigger_type="first_confluence",
             trigger_idx=3, start_idx=5, seq=0)
        return pool

    def test_est_established_after_the_window_is_excluded(self):
        from engine_v2.multitf.entity_df_mutation import _build_sibling_cts_ref_zone_from_pool
        with _patched_cts_derivation():
            zone = _build_sibling_cts_ref_zone_from_pool(
                self._pool(), "confluence", 0, 2, -1, (5, 24), None)
        assert zone is None          # anchor 20 is inside [5, 24]; the moment 26 is not

    def test_est_established_inside_the_window_is_read_at_its_anchor(self):
        from engine_v2.multitf.entity_df_mutation import _build_sibling_cts_ref_zone_from_pool
        with _patched_cts_derivation():
            zone = _build_sibling_cts_ref_zone_from_pool(
                self._pool(), "confluence", 0, 2, -1, (5, 26), None)
        assert zone is not None and zone.anchor_idx == 20

    def test_record_window_clips_on_the_moment(self):
        """The record's own window ends at 24 (replaced there); the read window
        runs to 30. The EST anchored at 20 but established at 26 belongs to the
        record after its end → excluded by the per-record clip (the read window
        alone would admit it)."""
        from engine_v2.multitf.entity_df_mutation import _build_sibling_cts_ref_zone_from_pool
        from engine_v2.tests._event_factory import make_cts_established
        pool = SubStructurePool()
        est = make_cts_established(cts_anchor_idx=20, confirmed_at=26, price=0.6020,
                                   structure_id=0, cycle_id=0, struct_direction=1)
        sib = _stub_sub(pool, direction=1, starting_idx=5, slice_begin=0, events=[est])
        _rec(sib, pool=pool, lens="confluence", S=0, C=2, tss=0, trigger_type="first_confluence",
             trigger_idx=3, start_idx=5, trigger_end_idx=24, end_reason="same_dir_replacement",
             seq=0)
        with _patched_cts_derivation():
            zone = _build_sibling_cts_ref_zone_from_pool(pool, "confluence", 0, 2, -1, (5, 30), None)
        assert zone is None

    def test_mirror_keeps_the_raw_idx_of_a_pattern_path_update(self):
        """(E3b landing review.) The mirror shifts the RAW idx (the moment since Plan
        E E4c) and `cts_anchor_idx` by one offset while the clip uses the moment: a
        pattern-path CTS_UPDATED anchored at 17 (moment 19), slice_begin 3 → the
        zone / probe input is the anchor 20, not the moment 22."""
        from engine_v2.multitf.entity_df_mutation import _build_sibling_cts_ref_zone_from_pool
        from engine_v2.tests._event_factory import make_event
        pool = SubStructurePool()
        upd = make_event("CTS_UPDATED", 17, price=0.6020, via="continuous", confirmed_at=19,
                         structure_id=0, cycle_id=0, struct_direction=1)
        sib = _stub_sub(pool, direction=1, starting_idx=5, slice_begin=3, events=[upd])
        _rec(sib, pool=pool, lens="confluence", S=0, C=2, tss=0, trigger_type="first_confluence",
             trigger_idx=3, start_idx=5, seq=0)
        with _patched_cts_derivation():
            zone = _build_sibling_cts_ref_zone_from_pool(pool, "confluence", 0, 2, -1, (5, 30), None)
        assert zone is not None and zone.anchor_idx == 20   # anchor 17 + slice_begin 3
