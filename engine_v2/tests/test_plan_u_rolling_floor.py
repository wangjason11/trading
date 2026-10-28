"""PLAN_U U2 — the rolling pinbar body floor (2026-10-28; the user: "the floor should be the most recent 90 pinbar's
p90", the history fetched before the frame — CANDLE_BODY_FLOOR_NOTES "Current logic").

Pinned here: the floor function (`features/pinbar_body_floor.py` — the window strictly before, linear interpolation,
the natural vs final membership, no full window, the seed), its frame wiring (`compute_candle_features`: the seed
fills only the windows; the one demotion predicate, asserted), the seed guard (`run_replay.fetch_floor_seed` /
`fetch_floor_seeds` with a stubbed `get_history`) and the pipeline wiring (`run_pipeline(floor_seeds=…)` reaches the H1
and the M15 classification). Each test names the mutant it kills (`qa_pinbar_floor_inputs/mutants_u2.py`).
"""
from __future__ import annotations

import contextlib
import dataclasses
import io
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from engine_v2.features.candle_classifier import apply_candle_classification
from engine_v2.features.candle_params import CandleParams
from engine_v2.features.candles_v2 import compute_candle_features
from engine_v2.features.pinbar_body_floor import pinbar_body_floor_for_frame, rolling_pinbar_body_floor

PIP = 0.0001


def _p(window=3, pct=50.0, source="natural", **over):
    return dataclasses.replace(CandleParams(), pinbar_floor_window=window, pinbar_floor_percentile=pct,
                               pinbar_floor_window_source=source, **over)


# --- candles: o = 0.6 each; a pinbar has a body of 20 % of its range, a maru 90 %, a normal 50 % -----------------

def _candle(kind, body_pips):
    b = body_pips * PIP
    r = b / {"P": 0.2, "M": 0.9, "N": 0.5}[kind]
    o = 0.6
    c = o + b
    if kind == "P":       # the body at the bottom of a long upper wick
        return {"o": o, "h": o + r - 0.0 * b, "l": o, "c": c}
    w = (r - b) / 2
    return {"o": o, "h": c + w, "l": o - w, "c": c}


def _raw(specs, start="2026-01-05"):
    df = pd.DataFrame([_candle(k, b) for k, b in specs])
    df.insert(0, "time", pd.date_range(start, periods=len(df), freq="h", tz="UTC"))
    df["volume"] = 1.0
    df.attrs["pair"] = "NZD_USD"
    return df


def _floors(specs, params, seed=None):
    return pinbar_body_floor_for_frame(_raw(specs), seed, params)


# --- the floor function -------------------------------------------------------------------------------------------

def test_the_settings_are_the_users():
    """The user's rule (2026-10-27 / 28): the p90 of the 90 most recent pinbars, a seed of 2 windows; the window =
    the FINAL pinbars (body-% + floor-demoted) — the user's pick at U3 (2026-10-28), natural (U2) kept as a switch."""
    p = CandleParams()
    assert (p.pinbar_floor_window, p.pinbar_floor_percentile, p.pinbar_floor_window_source,
            p.pinbar_floor_seed_windows) == (90, 90.0, "final", 2)


def test_the_window_is_strictly_before_the_candle():
    """Window 3: the 3rd pinbar has no floor (2 before it), the 4th candle's floor = the median of the 3 — mutants
    `u2_window_includes_candle` (the 3rd pinbar would get a floor) and `u2_window_skips_prev` (the 4th's would be NaN)."""
    floor, _ = _floors([("P", 1.0), ("P", 3.0), ("P", 2.0), ("M", 9.0), ("M", 9.0)], _p())
    assert np.isnan(floor[:3]).all()
    assert floor[3] == pytest.approx(2.0 * PIP) and floor[4] == pytest.approx(2.0 * PIP)


def test_the_window_rolls_to_the_most_recent_pinbars():
    """The 4th pinbar pushes the 1st out: [3, 2, 7] → median 3 — mutant `u2_window_hardcoded_90` (no full window at
    all)."""
    floor, _ = _floors([("P", 1.0), ("P", 3.0), ("P", 2.0), ("P", 7.0), ("M", 9.0)], _p())
    assert floor[3] == pytest.approx(2.0 * PIP) and floor[4] == pytest.approx(3.0 * PIP)


def test_the_percentile_is_numpy_linear():
    """Window 4 at p90 over [1, 2, 3, 4] pips: linear = 3.7 (the virtual index 2.7); nearest-rank would say 4 —
    mutants `u2_nearest_rank` and `u2_percentile_hardcoded_90` (p50 → 2.5 here, checked below)."""
    floor, _ = _floors([("P", 1.0), ("P", 2.0), ("P", 3.0), ("P", 4.0), ("M", 9.0)], _p(window=4, pct=90.0))
    assert floor[4] == pytest.approx(3.7 * PIP)
    floor50, _ = _floors([("P", 1.0), ("P", 2.0), ("P", 3.0), ("P", 4.0), ("M", 9.0)], _p(window=4, pct=50.0))
    assert floor50[4] == pytest.approx(2.5 * PIP)


_DEMOTE = [("P", 4.0), ("P", 4.0), ("P", 4.0), ("N", 1.0), ("M", 9.0)]   # the normal of 1 pip < the 4-pip floor


def test_natural_and_final_membership():
    """The demoted normal (1 pip) enters only the FINAL window: natural [4, 4, 4] → 4; final [4, 4, 1] → 4 at p50, so
    p0 (the minimum) shows it: natural 4, final 1 — mutant `u2_final_as_natural`."""
    nat, nat_dem = _floors(_DEMOTE, _p(pct=0.0, source="natural"))
    fin, fin_dem = _floors(_DEMOTE, _p(pct=0.0, source="final"))
    assert list(nat_dem) == list(fin_dem) == [False, False, False, True, False]
    assert nat[4] == pytest.approx(4.0 * PIP) and fin[4] == pytest.approx(1.0 * PIP)


def test_an_unknown_window_source_raises():
    with pytest.raises(ValueError, match="pinbar_floor_window_source"):
        _floors(_DEMOTE, _p(source="body"))


def test_the_function_uses_the_shared_predicate_with_eps():
    """A maru 1e-15 under its floor (float noise) is not demoted by the function; one 1e-11 under is — the window's
    membership reads `floor_demotes` (EPS 1e-12), like step 4 — mutant `u2_function_no_eps` (its own `<` without
    EPS). Exact arrays: a frame's bodies (c - o) carry their own float noise, which can land a body exactly AT the
    floor and hide the EPS."""
    floor, demoted = rolling_pinbar_body_floor([2e-4, 4e-4, 3e-4 - 1e-15, 3e-4 - 1e-11],
                                               ["pinbar", "pinbar", "maru", "maru"], _p(window=2, pct=50.0))
    assert floor[2] == floor[3] and floor[2] > 3e-4 - 1e-15
    assert list(demoted) == [False, False, False, True]


def test_compute_candle_features_asserts_the_demotions_agree(monkeypatch):
    """The floor function's own demotions must equal step 4's (`is_floor_pinbar`) — PLAN_U §8 A1; a mismatch fails
    loudly — mutant `u2_assert_dropped`."""
    import engine_v2.features.pinbar_body_floor as pbf
    real = pbf.pinbar_body_floor_for_frame

    def skewed(frame, seed, params):
        floor, demoted = real(frame, seed, params)
        return floor, ~demoted

    monkeypatch.setattr(pbf, "pinbar_body_floor_for_frame", skewed)
    with pytest.raises(AssertionError, match="demotions differ"):
        compute_candle_features(_raw(_DEMOTE), _p(), anchor_shifts=(0,))


# --- the seed -----------------------------------------------------------------------------------------------------

def test_a_seed_fills_the_first_windows_and_only_the_windows():
    """A seed of 3 pinbars gives the frame's FIRST candle a floor; the frame's prior-maru pool, big flags and volume
    stay frame-only (the seed's marus are not in candle 0's pool) — mutant `u2_seed_into_maru_pool`."""
    seed = _raw([("M", 30.0), ("P", 2.0), ("P", 2.0), ("P", 2.0)], start="2026-01-04")
    frame = _raw([("N", 1.0), ("M", 9.0)])
    out = compute_candle_features(frame, _p(), anchor_shifts=(0,), floor_seed=seed)
    assert out.loc[0, "pinbar_body_floor"] == pytest.approx(2.0 * PIP) and out.loc[0, "candle_type"] == "pinbar"
    assert bool(out.loc[0, "is_floor_pinbar"]) is True
    assert out.loc[0, "prior_maru_max_len_as0"] == 0.0 and out.loc[1, "prior_maru_max_len_as0"] == 0.0
    no_seed = compute_candle_features(frame, _p(), anchor_shifts=(0,))
    assert no_seed["pinbar_body_floor"].isna().all() and no_seed.loc[0, "candle_type"] == "normal"


def test_natural_seeded_equals_frame_only_where_both_have_a_window():
    """Natural mode: a candle's floor depends only on its window, so the seeded frame = the frame-only one wherever the
    frame-only run has a full window (the measured check — PLAN_U §5.3; not so in final mode, §8 A2)."""
    rng = np.random.default_rng(7)
    kinds = rng.choice(["P", "M", "N"], size=120)
    specs = [(k, float(rng.integers(1, 30)) / 4) for k in kinds]
    seed = _raw(specs[:30], start="2025-12-30")
    frame = _raw(specs[30:])
    seeded, _ = pinbar_body_floor_for_frame(frame, seed, _p(window=5, pct=90.0))
    alone, _ = pinbar_body_floor_for_frame(frame, None, _p(window=5, pct=90.0))
    both = ~np.isnan(alone)
    assert both.sum() > 20 and np.array_equal(seeded[both], alone[both])
    assert not np.isnan(seeded).any()


def test_the_seeds_demotions_count_in_the_final_window():
    """Final mode: a demotion INSIDE the seed enters the frame's first window (p0 → 1 pip); natural mode ignores it —
    mutant `u2_seed_demotions_dropped` (the window pre-filled with the seed's body-% pinbars only)."""
    seed = _raw(_DEMOTE[:4], start="2026-01-04")          # 3 pinbars of 4 pips, then a 1-pip normal (demoted)
    frame = _raw([("M", 9.0)])
    fin, _ = pinbar_body_floor_for_frame(frame, seed, _p(pct=0.0, source="final"))
    nat, _ = pinbar_body_floor_for_frame(frame, seed, _p(pct=0.0, source="natural"))
    assert fin[0] == pytest.approx(1.0 * PIP) and nat[0] == pytest.approx(4.0 * PIP)


def test_a_seed_not_strictly_before_the_frame_fails_loudly():
    seed = _raw([("P", 2.0)], start="2026-01-05")          # its candle AT the frame's first time
    with pytest.raises(AssertionError, match="not strictly before"):
        pinbar_body_floor_for_frame(_raw([("M", 9.0)]), seed, _p())


def test_apply_candle_classification_and_prepare_lower_tf_data_pass_the_seed():
    """The seed reaches the classifier through both frame entry points — mutant `u2_bridge_drops_seed`."""
    from engine_v2.multitf.data_bridge import prepare_lower_tf_data
    seed = _raw([("P", 2.0)] * 90, start="2025-12-01")
    frame = _raw([("N", 1.0), ("M", 9.0)])
    h1 = apply_candle_classification(frame, floor_seed=seed).df
    m15 = prepare_lower_tf_data(frame, floor_seed=seed)
    for df in (h1, m15):
        assert list(df["pinbar_body_floor"]) == pytest.approx([2.0 * PIP] * 2)
        assert df.loc[0, "candle_type"] == "pinbar"


# --- the seed guard (`run_replay.fetch_floor_seed`, a stubbed OANDA) ------------------------------------------------

FIRST = pd.Timestamp("2026-01-05 00:00", tz="UTC")


class _Oanda:
    """`get_history` over a synthetic H1 history: one candle per hour before `FIRST` (and one AT it — OANDA's `to` is
    exclusive, the stub is not), every `pin_every`-th a pinbar; records each request; `fail` = how many first calls
    raise (a retried 504)."""

    def __init__(self, pin_every=2, fail=0, empty_until=None):
        self.calls, self.pin_every, self.fail, self.empty_until = [], pin_every, fail, empty_until

    def __call__(self, *, pair, timeframe, start, end):
        self.calls.append((pd.Timestamp(start), pd.Timestamp(end)))
        if self.fail:
            self.fail -= 1
            raise RuntimeError("OANDA candles request failed 504: gateway")
        if self.empty_until is not None and len(self.calls) <= self.empty_until:
            return pd.DataFrame(columns=["time", "o", "h", "l", "c", "volume"])
        times = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="h", tz="UTC")   # inclusive of `end`
        k = ((times - pd.Timestamp("2000-01-01", tz="UTC")) // pd.Timedelta(hours=1)).to_numpy()
        rows = [_candle("P" if i % self.pin_every == 0 else "M", 2.0) for i in k]
        df = pd.DataFrame(rows)
        df.insert(0, "time", times)
        df["volume"] = 1.0
        return df


@pytest.fixture
def oanda(monkeypatch):
    import engine_v2.run_replay as rr

    def make(**kw):
        stub = _Oanda(**kw)
        monkeypatch.setattr(rr, "get_history", stub)
        monkeypatch.setattr(rr.time, "sleep", lambda s: None)
        return stub
    return make


def test_the_guard_steps_back_until_the_seed_holds_enough_pinbars(oanda, capsys):
    """400-candle steps, the whole span re-fetched per step, until >= min_pinbars body-% pinbars: 300 needs 2 steps
    (step 1: 200 pinbars in 400 h) — the spans, the stop, the line."""
    from engine_v2.run_replay import fetch_floor_seed
    stub = oanda(pin_every=2)
    seed = fetch_floor_seed("NZD_USD", "H1", FIRST, min_pinbars=300)
    assert stub.calls == [(FIRST - timedelta(hours=400 * k), FIRST) for k in (1, 2)]
    assert len(seed) == 800 and seed["time"].iloc[-1] < FIRST
    assert "[floor_seed] H1: 800 candles 2025-12-02 16:00:00+00:00 -> 2026-01-04 23:00:00+00:00, 2 steps, " \
           "400 body-% pinbars (need 300)" in capsys.readouterr().out


def test_the_guard_stops_at_exactly_enough(oanda):
    """Step 1 holds exactly 200 pinbars = min_pinbars → stop (>=) — mutant `u2_guard_strict_gt` (a second step)."""
    from engine_v2.run_replay import fetch_floor_seed
    stub = oanda(pin_every=2)
    fetch_floor_seed("NZD_USD", "H1", FIRST, min_pinbars=200)
    assert len(stub.calls) == 1


def test_the_seed_drops_the_candle_at_the_frames_first_time(oanda):
    """The stub returns a candle AT `first_time` too; it is dropped — mutant `u2_seed_no_time_filter` (the assert
    after the filter, or the seed overlapping the frame)."""
    from engine_v2.run_replay import fetch_floor_seed
    oanda(pin_every=2)
    seed = fetch_floor_seed("NZD_USD", "H1", FIRST, min_pinbars=10)
    assert len(seed) == 400 and (pd.to_datetime(seed["time"], utc=True) < FIRST).all()


def test_the_guard_raises_after_max_steps(oanda):
    from engine_v2.run_replay import fetch_floor_seed
    stub = oanda(pin_every=1000)
    with pytest.raises(RuntimeError, match=r"^\[floor_seed\] ERROR: H1 NZD_USD: 4 body-% pinbars in 3200 candles"):
        fetch_floor_seed("NZD_USD", "H1", FIRST, min_pinbars=180, max_steps=8)
    assert len(stub.calls) == 8


def test_an_empty_step_is_not_an_error(oanda):
    """A span with no candle holds no pinbar: the guard steps on — mutant `u2_seed_empty_raises`."""
    from engine_v2.run_replay import fetch_floor_seed
    stub = oanda(pin_every=2, empty_until=1)
    seed = fetch_floor_seed("NZD_USD", "H1", FIRST, min_pinbars=300)
    assert len(stub.calls) == 2 and len(seed) == 800


def test_a_failed_seed_request_retries_under_its_own_prefix(oanda, capsys):
    """Never `[data_bridge]` / `[multi_tf:dual]` / `[auto_extend]` — the fetch gate keys on those — mutant
    `u2_seed_prefix`."""
    from engine_v2.run_replay import fetch_floor_seed
    oanda(pin_every=2, fail=1)
    fetch_floor_seed("NZD_USD", "H1", FIRST, min_pinbars=10)
    out = capsys.readouterr().out
    assert "[floor_seed] RETRY 1/2 H1 fetch" in out
    assert "[auto_extend]" not in out and "[data_bridge]" not in out and "[multi_tf:dual]" not in out


def test_a_seed_request_that_keeps_failing_raises(oanda):
    from engine_v2.run_replay import fetch_floor_seed
    oanda(pin_every=2, fail=3)
    with pytest.raises(RuntimeError, match=r"^\[floor_seed\] ERROR fetching H1"):
        fetch_floor_seed("NZD_USD", "H1", FIRST, min_pinbars=10)


def test_the_steps_stay_under_oandas_candle_limit():
    from engine_v2.run_replay import fetch_floor_seed
    with pytest.raises(AssertionError, match="5000"):
        fetch_floor_seed("NZD_USD", "H1", FIRST, min_pinbars=10, step_candles=1000, max_steps=8)


def test_the_replay_fetches_both_seeds_for_two_windows(oanda, capsys):
    """`fetch_floor_seeds` (what `main()` calls): the H1 seed and, with M15 on, the M15 seed, each for
    seed_windows x window = 180 pinbars, both before the H1 frame's first candle — mutant `u2_seed_one_window`."""
    from engine_v2.run_replay import fetch_floor_seeds
    stub = oanda(pin_every=2)
    seeds = fetch_floor_seeds("NZD_USD", "H1", FIRST, ("M15",))
    assert set(seeds) == {"H1", "M15"} and all(c[1] == FIRST for c in stub.calls)
    out = capsys.readouterr().out
    assert out.count("(need 180)") == 2
    assert set(fetch_floor_seeds("NZD_USD", "H1", FIRST, ())) == {"H1"}


# --- the pipeline wiring ------------------------------------------------------------------------------------------

class _Stop(Exception):
    pass


def test_run_pipeline_passes_the_h1_seed(monkeypatch):
    """`run_pipeline(floor_seeds=…)` → the H1 classification gets `floor_seeds["H1"]` — mutant `u2_pipeline_no_h1_seed`."""
    import engine_v2.pipeline.orchestrator as orch
    import engine_v2.tests.test_render_sub_projection as RSP
    seen = {}

    def spy(df, **kw):
        seen.update(kw)
        raise _Stop

    monkeypatch.setattr(orch, "apply_candle_classification", spy)
    seed = pd.DataFrame({"time": [0]})
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(_Stop):
        orch.run_pipeline(RSP._make_raw_df(RSP._make_flat_prefix()), floor_seeds={"H1": seed, "M15": None})
    assert seen["floor_seed"] is seed


def test_run_pipeline_passes_the_m15_seed_to_the_m15_build(monkeypatch):
    """`run_pipeline(lower_timeframes=("M15",), floor_seeds=…)` → `_run_multi_tf_dual(m15_floor_seed=floor_seeds["M15"])`
    — mutant `u2_pipeline_no_m15_seed`."""
    import engine_v2.pipeline.orchestrator as orch
    import engine_v2.tests.test_render_sub_projection as RSP
    seen = {}

    def spy(*a, **kw):
        seen.update(kw)
        raise _Stop

    monkeypatch.setattr(orch, "_run_multi_tf_dual", spy)
    seed = pd.DataFrame({"time": [0]})
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(_Stop):
        orch.run_pipeline(RSP._make_raw_df(RSP._make_flat_prefix() + RSP._make_reversing_data()),
                          lower_timeframes=("M15",), floor_seeds={"M15": seed})
    assert seen["m15_floor_seed"] is seed


def test_run_pipeline_rejects_an_unknown_seed_timeframe():
    import engine_v2.pipeline.orchestrator as orch
    import engine_v2.tests.test_render_sub_projection as RSP
    with pytest.raises(AssertionError, match="floor_seeds"):
        orch.run_pipeline(RSP._make_raw_df(RSP._make_flat_prefix()), floor_seeds={"H4": None})


def test_the_m15_build_passes_the_m15_seed(monkeypatch):
    """`_run_multi_tf_dual(m15_floor_seed=…)` → `prepare_lower_tf_data(…, floor_seed=…)` — mutant
    `u2_dual_no_m15_seed`. The collaborators are imported INSIDE the function: patched on their modules."""
    from types import SimpleNamespace
    from engine_v2.pipeline.orchestrator import _run_multi_tf_dual
    seen = {}

    def prep(df, **kw):
        seen.update(kw)
        raise _Stop

    monkeypatch.setattr("engine_v2.multitf.first_confluence_pipeline.to_multi_tf_trigger",
                        lambda _v, _h1: SimpleNamespace(parent_sid=0, parent_cycle_id=1, lower_sd=+1))
    monkeypatch.setattr("engine_v2.multitf.uc1_trigger.detect_uc1_triggers", lambda *a, **k: [])
    monkeypatch.setattr("engine_v2.multitf.data_bridge.fetch_lower_tf_data", lambda *a, **k: pd.DataFrame({"time": [0]}))
    monkeypatch.setattr("engine_v2.multitf.data_bridge.prepare_lower_tf_data", prep)
    h1 = pd.DataFrame({"time": pd.date_range("2025-11-17", periods=40, freq="h", tz="UTC")})
    v1 = SimpleNamespace(status="finalized", trigger_event_idx=10, input_idx=3, parent_cts_anchor_idx=7)
    seed = pd.DataFrame({"time": [0]})
    with pytest.raises(_Stop):
        _run_multi_tf_dual(h1, [], [], [], {}, None, first_confluence_triggers=[v1], m15_floor_seed=seed)
    assert seen["floor_seed"] is seed
