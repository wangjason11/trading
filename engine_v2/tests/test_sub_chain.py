"""Tests for the Phase 2 merge-and-bound subordinate sub-build
(Part 4 §6.1, REVISED 2026-05-25).

Two layers:

  * ``TestBuildOneSid`` — integration of ``build_one_sid`` on real prepared
    fixtures (reuses the self-contained candle builders from
    test_bounded_structure). Pins: one bounded sid mirrors with the canonical
    identity; reversal reports the handoff; the start_idx>=n / degenerate
    guards return None; sequential sids on one entity df carry NO cascade tag.

  * ``TestBuildParentCycleChain`` — control-logic of the chain driver with the
    heavy pieces (probe+map, lifecycle-end, build_one_sid) stubbed, so sid
    sequencing / reversal-vs-subsequent boundary races / per-cycle sid reset /
    pending-bootstrap are deterministic.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from engine_v2.multitf import entity_df_mutation as edm
from engine_v2.multitf.entity_df_mutation import (
    SidBuildOutcome,
    build_one_sid,
    build_parent_cycle_chain,
)
from engine_v2.multitf.types import MultiTFTrigger
from engine_v2.features.candle_classifier import apply_candle_classification
from engine_v2.patterns.pattern_engine import detect_patterns
from engine_v2.patterns.imbalance import compute_imbalance


# ---------------------------------------------------------------------------
# Fixtures (mirror test_bounded_structure — self-contained)
# ---------------------------------------------------------------------------

def _make_raw_df(ohlc_rows: list[dict], pair: str = "NZD_USD") -> pd.DataFrame:
    base_time = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    records = []
    for i, row in enumerate(ohlc_rows):
        records.append({
            "time": base_time + pd.Timedelta(minutes=15 * i),
            "o": row["o"], "h": row["h"], "l": row["l"], "c": row["c"],
            "volume": row.get("volume", 100),
        })
    df = pd.DataFrame(records)
    df.attrs["pair"] = pair
    return df


def _prepare_df(ohlc_rows: list[dict], pair: str = "NZD_USD") -> pd.DataFrame:
    raw = _make_raw_df(ohlc_rows, pair=pair)
    c_res = apply_candle_classification(raw)
    p_res = detect_patterns(c_res.df)
    df = compute_imbalance(p_res.df)
    df.attrs["pair"] = pair
    return df


def _make_uptrend_data(n: int = 80, base: float = 0.6000,
                       step: float = 0.0020, noise: float = 0.0005) -> list[dict]:
    rows = []
    rng = np.random.RandomState(42)
    price = base
    for _ in range(n):
        body = step + rng.uniform(0, noise)
        o = price
        c = price + body
        h = c + rng.uniform(0.0001, 0.0005)
        l = o - rng.uniform(0.0001, 0.0005)
        rows.append({"o": round(o, 5), "h": round(h, 5),
                     "l": round(l, 5), "c": round(c, 5)})
        price = c
    return rows


def _seg(rows, price, n, direction, step, noise, rng):
    for _ in range(n):
        body = step + rng.uniform(0, noise)
        o = price
        c = price + direction * body
        if direction > 0:
            h = c + rng.uniform(0.0001, 0.0004)
            l = o - rng.uniform(0.0001, 0.0004)
        else:
            h = o + rng.uniform(0.0001, 0.0004)
            l = c - rng.uniform(0.0001, 0.0004)
        rows.append({"o": round(o, 5), "h": round(h, 5),
                     "l": round(l, 5), "c": round(c, 5)})
        price = c
    return price


def _make_reversing_data() -> list[dict]:
    """Up impulses with pullbacks, then a strong down leg → reversal ~idx 52
    (validated in test_bounded_structure)."""
    rng = np.random.RandomState(7)
    rows: list[dict] = []
    price = 0.6000
    for _ in range(3):
        price = _seg(rows, price, 6, +1, 0.0020, 0.0004, rng)
        price = _seg(rows, price, 3, -1, 0.0008, 0.0003, rng)
    price = _seg(rows, price, 6, +1, 0.0020, 0.0004, rng)
    price = _seg(rows, price, 20, -1, 0.0022, 0.0004, rng)
    price = _seg(rows, price, 8, -1, 0.0006, 0.0003, rng)
    return rows


def _mt(use_case="first_confluence", parent_sid=0, parent_cycle_id=0,
        lower_sd=1, trigger_event_idx=0, m15_start=0) -> MultiTFTrigger:
    return MultiTFTrigger(
        parent_tf="H1",
        parent_sid=parent_sid,
        parent_cycle_id=parent_cycle_id,
        parent_sd=lower_sd,
        use_case=use_case,
        lower_tf="M15",
        lower_sd=lower_sd,
        start_time=pd.Timestamp("2024-01-01", tz="UTC"),
        start_price=0.6,
        lifecycle_end_idx=None,
        meta={"trigger_event_idx": trigger_event_idx, "_test_start": m15_start},
    )


_PATH = "H1.main >> M15.confluence"


# ---------------------------------------------------------------------------
# Layer 1 — build_one_sid integration (real fixtures)
# ---------------------------------------------------------------------------

class TestBuildOneSid:
    def test_no_reversal_mirrors_with_identity(self):
        df = _prepare_df(_make_uptrend_data(n=80))
        trig = _mt()
        out = build_one_sid(
            df,
            start_m15_abs=0, sd=1, end_m15_abs=int(df.index[-1]),
            sub_path_id=_PATH, timeframe="M15", trigger=trig,
            sub_sid=0, started_by="first_confluence",
            start_trigger_idx=0,
        )
        assert out is not None
        assert out.reversal_idx_abs is None
        assert out.next_start_abs is None
        assert out.result.meta["end_reason"] == "lifecycle_end"

        # Identity is stamped on mirrored snapshots.
        events = df.attrs.get("events", [])
        assert events, "expected mirrored events"
        ev = events[0]
        assert ev.meta["sub_sid"] == 0
        assert ev.meta["started_by"] == "first_confluence"
        assert ev.meta["start_trigger_idx"] == 0
        assert ev.meta["parent_sid"] == 0
        assert ev.meta["parent_cycle_id"] == 0

    def test_reversal_reports_handoff(self):
        df = _prepare_df(_make_reversing_data())
        trig = _mt(lower_sd=1)
        out = build_one_sid(
            df,
            start_m15_abs=0, sd=1, end_m15_abs=int(df.index[-1]),
            sub_path_id=_PATH, timeframe="M15", trigger=trig,
            sub_sid=0, started_by="first_confluence",
            start_trigger_idx=0,
        )
        assert out is not None
        assert out.reversal_idx_abs is not None and out.reversal_idx_abs > 10
        # scenario-2 handoff for the next (reversal-born) sid. The handoff
        # start is the post-reversal structural anchor (a swing extreme),
        # which may precede the reversal *confirmation* idx — only require a
        # valid in-range idx + the flipped direction.
        assert out.next_start_abs is not None
        assert 0 <= out.next_start_abs < len(df)
        assert out.next_sd == -1            # flipped from the +1 run
        assert out.result.meta["end_reason"] == "reversal"

    def test_start_past_end_of_data_returns_none(self):
        """start_idx >= n guard (LANDMINE) — no AttributeError."""
        df = _prepare_df(_make_uptrend_data(n=60))
        n = len(df)
        out = build_one_sid(
            df,
            start_m15_abs=n, sd=1, end_m15_abs=n + 5,
            sub_path_id=_PATH, timeframe="M15", trigger=_mt(),
            sub_sid=0, started_by="reversal", start_trigger_idx=n,
        )
        assert out is None

    def test_degenerate_window_returns_none(self):
        df = _prepare_df(_make_uptrend_data(n=60))
        out = build_one_sid(
            df,
            start_m15_abs=30, sd=1, end_m15_abs=30,   # start >= end
            sub_path_id=_PATH, timeframe="M15", trigger=_mt(),
            sub_sid=0, started_by="first_confluence",
            start_trigger_idx=30,
        )
        assert out is None

    def test_sequential_sids_no_cascade_tag(self):
        """Two non-overlapping sids on one entity df → no overwrite cascade."""
        df = _prepare_df(_make_uptrend_data(n=80))
        out0 = build_one_sid(
            df, start_m15_abs=0, sd=1, end_m15_abs=39,
            sub_path_id=_PATH, timeframe="M15", trigger=_mt(),
            sub_sid=0, started_by="first_confluence",
            start_trigger_idx=0,
        )
        out1 = build_one_sid(
            df, start_m15_abs=40, sd=1, end_m15_abs=79,
            sub_path_id=_PATH, timeframe="M15",
            trigger=_mt(use_case="subsequent_confluence"),
            sub_sid=1, started_by="subsequent_confluence",
            start_trigger_idx=40,
        )
        assert out0 is not None and out1 is not None

        for key in ("kl_zones", "poi_zones", "fib_states", "wvmi",
                    "events"):
            for snap in df.attrs.get(key, []):
                meta = snap.meta if hasattr(snap, "meta") else {}
                dby = str(meta.get("deactivated_by", ""))
                assert not dby.startswith("overwritten_by_sid"), (
                    f"{key} snapshot carries a cascade tag {dby!r} — the "
                    f"merge-and-bound build must never fire the cascade"
                )

        # Both sids' events present, distinct sub_sids (same parent cycle).
        sub_sids = {ev.meta.get("sub_sid") for ev in df.attrs.get("events", [])}
        assert {0, 1} <= sub_sids


# ---------------------------------------------------------------------------
# Layer 2 — build_parent_cycle_chain control logic (stubbed heavy pieces)
# ---------------------------------------------------------------------------

def _dummy_df(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
        "h": np.linspace(0.60, 0.62, n),
        "l": np.linspace(0.59, 0.61, n),
    })


def _install_stubs(monkeypatch, *, cycle_end, reversals=None):
    """Patch probe+map, lifecycle-end, and build_one_sid for the driver.

    `reversals` maps a per-cycle `sub_sid` -> (reversal_idx_abs,
    next_start_abs, next_sd); absent sids produce no reversal. Returns `calls`
    (list of dicts).
    """
    reversals = reversals or {}
    calls: list[dict] = []

    def fake_resolve(trigger, parent_df, entity_df):
        return trigger.meta.get("_test_start"), 0

    def fake_map(parent_idx, parent_df, m15_df):
        return int(parent_idx)   # identity: trigger_event_idx IS the boundary

    def fake_cycle_end(trigger, m15_df, parent_df):
        return cycle_end

    def fake_build_one_sid(entity_df, *, start_m15_abs, sd, end_m15_abs,
                           sub_path_id, timeframe, trigger, sub_sid,
                           started_by, start_trigger_idx,
                           validated_parent_idx=None, parent_floor_m15=None):
        calls.append({
            "start": start_m15_abs, "sd": sd, "end": end_m15_abs,
            "sub_sid": sub_sid, "started_by": started_by,
            "start_trigger_idx": start_trigger_idx,
        })
        rev = reversals.get(sub_sid, (None, None, None))
        result = SimpleNamespace(
            trigger=trigger,
            meta={"sub_sid": sub_sid},
            wvmi_records=[],
        )
        return SidBuildOutcome(
            result=result, reversal_idx_abs=rev[0],
            next_start_abs=rev[1], next_sd=rev[2],
        )

    monkeypatch.setattr(edm, "_resolve_trigger_m15_start", fake_resolve)
    monkeypatch.setattr(edm, "_map_parent_idx_to_m15_hour_end", fake_map)
    monkeypatch.setattr(edm, "build_one_sid", fake_build_one_sid)
    monkeypatch.setattr(
        "engine_v2.multitf.lower_tf_pipeline._find_m15_lifecycle_end",
        fake_cycle_end,
    )
    return calls


class TestBuildParentCycleChain:
    def test_bootstrap_only(self, monkeypatch):
        calls = _install_stubs(monkeypatch, cycle_end=80)
        boot = _mt(m15_start=10)
        results = build_parent_cycle_chain(
            _dummy_df(100), _dummy_df(100),
            bootstrap=boot, subsequents=[], sub_path_id=_PATH,
        )
        assert len(results) == 1
        assert len(calls) == 1
        assert calls[0]["sub_sid"] == 0
        assert calls[0]["started_by"] == "first_confluence"
        assert calls[0]["end"] == 80          # bounded by cycle end (no sub)

    def test_bootstrap_plus_subsequent(self, monkeypatch):
        calls = _install_stubs(monkeypatch, cycle_end=80)
        boot = _mt(m15_start=10)
        sub = _mt(use_case="subsequent_confluence", trigger_event_idx=45,
                  m15_start=40)
        results = build_parent_cycle_chain(
            _dummy_df(100), _dummy_df(100),
            bootstrap=boot, subsequents=[sub], sub_path_id=_PATH,
        )
        assert len(calls) == 2
        # sub_sid 0 bounded AT the subsequent boundary.
        assert calls[0]["sub_sid"] == 0 and calls[0]["end"] == 45
        # sub_sid 1 = subsequent-born, starts at probe start.
        assert calls[1]["sub_sid"] == 1
        assert calls[1]["started_by"] == "subsequent_confluence"
        assert calls[1]["start"] == 40 and calls[1]["end"] == 80

    def test_reversal_born_sid(self, monkeypatch):
        calls = _install_stubs(
            monkeypatch, cycle_end=80,
            reversals={0: (40, 45, -1)},   # sub_sid 0 reverses at 40 → start 45, sd -1
        )
        boot = _mt(m15_start=10, lower_sd=1)
        results = build_parent_cycle_chain(
            _dummy_df(100), _dummy_df(100),
            bootstrap=boot, subsequents=[], sub_path_id=_PATH,
        )
        assert len(calls) == 2
        assert calls[1]["started_by"] == "reversal"
        assert calls[1]["start"] == 45 and calls[1]["sd"] == -1
        assert calls[1]["start_trigger_idx"] == 40

    def test_reversal_then_subsequent_both_build(self, monkeypatch):
        calls = _install_stubs(
            monkeypatch, cycle_end=80,
            reversals={0: (30, 35, -1)},   # sub_sid 0 reverses before the sub
        )
        boot = _mt(m15_start=10, lower_sd=1)
        sub = _mt(use_case="subsequent_confluence", trigger_event_idx=65,
                  m15_start=60)
        results = build_parent_cycle_chain(
            _dummy_df(100), _dummy_df(100),
            bootstrap=boot, subsequents=[sub], sub_path_id=_PATH,
        )
        assert len(calls) == 3
        assert [c["started_by"] for c in calls] == [
            "first_confluence", "reversal", "subsequent_confluence",
        ]
        assert [c["sub_sid"] for c in calls] == [0, 1, 2]
        # reversal-born sid is bounded by the still-pending subsequent.
        assert calls[1]["start"] == 35 and calls[1]["end"] == 65
        assert calls[2]["start"] == 60 and calls[2]["end"] == 80

    def test_reversal_handoff_past_end_of_data_guarded(self, monkeypatch):
        """next_start past end-of-data → reversal branch skipped, chain ends."""
        calls = _install_stubs(
            monkeypatch, cycle_end=49,
            reversals={0: (45, 55, -1)},   # next_start 55 >= len(entity_df) 50
        )
        results = build_parent_cycle_chain(
            _dummy_df(50), _dummy_df(50),
            bootstrap=_mt(m15_start=10), subsequents=[], sub_path_id=_PATH,
        )
        assert len(calls) == 1   # no crash, no extra sid

    def test_two_cycles_sub_sid_resets(self, monkeypatch):
        # sub_sid resets to 0 each parent cycle; the identity tuple differs
        # only by parent_cycle_id across the two cycles.
        _install_stubs(monkeypatch, cycle_end=40)
        rA = build_parent_cycle_chain(
            _dummy_df(100), _dummy_df(100),
            bootstrap=_mt(parent_cycle_id=0, m15_start=10), subsequents=[],
            sub_path_id=_PATH,
        )
        _install_stubs(monkeypatch, cycle_end=90)  # reset calls list for clarity
        rB = build_parent_cycle_chain(
            _dummy_df(100), _dummy_df(100),
            bootstrap=_mt(parent_cycle_id=1, m15_start=50), subsequents=[],
            sub_path_id=_PATH,
        )
        assert rA[0].meta["sub_sid"] == 0 and rA[0].trigger.parent_cycle_id == 0
        assert rB[0].meta["sub_sid"] == 0 and rB[0].trigger.parent_cycle_id == 1

    def test_pending_bootstrap_no_chain(self, monkeypatch):
        calls = _install_stubs(monkeypatch, cycle_end=80)
        boot = _mt(m15_start=None)   # fake_resolve returns (None, 0) → pending
        results = build_parent_cycle_chain(
            _dummy_df(100), _dummy_df(100),
            bootstrap=boot, subsequents=[_mt(use_case="subsequent_confluence",
                                             trigger_event_idx=45, m15_start=40)],
            sub_path_id=_PATH,
        )
        assert results == []
        assert len(calls) == 0
