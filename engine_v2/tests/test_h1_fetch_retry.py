"""The parent-TF (H1) fetch retries like an M15 chunk (2026-09-28; zones-audit "Still open": the H1 fetch retry).

`run_replay.fetch_history_with_auto_extend` makes one `get_history` request per auto-extend iteration (5 on the
reference window) and used to crash on the first non-200 — OANDA 504s stopped two replays on 2026-09-28. Every
request now goes through `run_replay._get_history_retrying`: the M15 policy (`data_bridge._FETCH_RETRIES` retries,
`_FETCH_RETRY_WAIT_S` apart, same catch set), one ``[auto_extend] RETRY k/N …`` line per retry, then a raise; an
empty frame raises too.

The `/compare` §2b fetch gate needs no change for these lines: a failed H1 fetch crashes the run before its timing
block (the gate's same-run check FAILs), and its N/A branch keys on the absence of `[data_bridge]` /
`[multi_tf:dual]` lines — so an H1 line must never carry either prefix.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from engine_v2 import run_replay
from engine_v2.multitf import data_bridge

START = datetime(2025, 12, 1, tzinfo=timezone.utc)
END = datetime(2026, 1, 20, tzinfo=timezone.utc)
_GATE_PREFIXES = ("[data_bridge]", "[multi_tf:dual]")


def _frame(n=3):
    return pd.DataFrame({
        "time": pd.date_range("2025-12-01", periods=n, freq="h", tz="UTC"),
        "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "volume": 1,
    })


class _FakeHistory:
    """`get_history` stand-in: `outcomes[k]` is the k-th request's outcome — a DataFrame (returned) or an exception
    (raised); the last outcome repeats."""

    def __init__(self, *outcomes):
        self.outcomes = outcomes
        self.starts = []   # the `start` of each request

    def __call__(self, *, pair, timeframe, start, end):
        self.starts.append(start)
        outcome = self.outcomes[min(len(self.starts) - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture
def sleeps(monkeypatch):
    waited = []
    monkeypatch.setattr(run_replay.time, "sleep", waited.append)
    return waited


@pytest.fixture
def start_ok(monkeypatch):
    """`identify_start_scenario_1` stand-in: never 'too early' (one request per fetch)."""
    monkeypatch.setattr(run_replay, "identify_start_scenario_1",
                        lambda df, **_k: SimpleNamespace(meta={"too_early": False}, start_idx=0))


def _http_504():
    # a gateway error body is multi-line HTML (`get_history` embeds `resp.text[:500]`)
    return RuntimeError("OANDA candles request failed 504: <html>\n<body>gateway timeout</body>\n</html>")


def _fetch(**kw):
    return run_replay.fetch_history_with_auto_extend(pair="NZD_USD", timeframe="H1", start=START, end=END, **kw)


def test_a_clean_fetch_prints_nothing_and_never_sleeps(monkeypatch, capsys, sleeps, start_ok):
    fake = _FakeHistory(_frame())
    monkeypatch.setattr(run_replay, "get_history", fake)
    df, eff_start, _ = _fetch()
    assert len(df) == 3 and eff_start == START
    assert fake.starts == [START] and sleeps == [] and capsys.readouterr().out == ""


def test_a_transient_failure_recovers_with_one_retry_line_each(monkeypatch, capsys, sleeps, start_ok):
    fake = _FakeHistory(_http_504(), requests.ConnectionError("reset"), _frame())
    monkeypatch.setattr(run_replay, "get_history", fake)
    df, _, _ = _fetch()
    assert len(df) == 3 and fake.starts == [START] * 3
    # the M15 policy, not a copy of it (a float literal compiled in run_replay would be a different object)
    assert run_replay._FETCH_RETRY_WAIT_S is data_bridge._FETCH_RETRY_WAIT_S
    assert sleeps == [data_bridge._FETCH_RETRY_WAIT_S] * data_bridge._FETCH_RETRIES and sleeps[0] > 0
    out = capsys.readouterr().out.splitlines()
    assert out == [
        "[auto_extend] RETRY 1/2 H1 fetch 2025-12-01 00:00:00+00:00-2026-01-20 00:00:00+00:00 in 5 s: "
        "OANDA candles request failed 504: <html> <body>gateway timeout</body> </html>",
        "[auto_extend] RETRY 2/2 H1 fetch 2025-12-01 00:00:00+00:00-2026-01-20 00:00:00+00:00 in 5 s: reset",
    ]
    assert not any(p in line for line in out for p in _GATE_PREFIXES)


def test_a_request_that_keeps_failing_raises_after_two_retries(monkeypatch, capsys, sleeps):
    def _never_called(*_a, **_k):
        raise AssertionError("no frame, no start search")

    monkeypatch.setattr(run_replay, "identify_start_scenario_1", _never_called)
    fake = _FakeHistory(_http_504())
    monkeypatch.setattr(run_replay, "get_history", fake)
    with pytest.raises(RuntimeError) as exc_info:
        _fetch()
    assert str(exc_info.value) == (
        "[auto_extend] ERROR fetching H1 2025-12-01 00:00:00+00:00-2026-01-20 00:00:00+00:00 after 2 retries: "
        "OANDA candles request failed 504: <html> <body>gateway timeout</body> </html>")
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert fake.starts == [START] * 3 and len(sleeps) == 2
    assert [line.split(" H1 fetch")[0] for line in capsys.readouterr().out.splitlines()] == [
        "[auto_extend] RETRY 1/2", "[auto_extend] RETRY 2/2"]


@pytest.mark.parametrize("error", [FileNotFoundError("Could not find 'oanda.cfg'"),   # creds
                                   KeyError("time"), ValueError("x"), TypeError("float(None)")],  # payload
                         ids=lambda e: type(e).__name__)
def test_a_non_transient_error_is_raised_at_once(monkeypatch, capsys, sleeps, start_ok, error):
    fake = _FakeHistory(error)
    monkeypatch.setattr(run_replay, "get_history", fake)
    with pytest.raises(type(error)):
        _fetch()
    assert fake.starts == [START] and sleeps == [] and capsys.readouterr().out == ""


def test_an_empty_frame_raises_and_is_not_retried(monkeypatch, capsys, sleeps, start_ok):
    fake = _FakeHistory(_frame(n=0))
    monkeypatch.setattr(run_replay, "get_history", fake)
    with pytest.raises(RuntimeError) as exc_info:
        _fetch()
    assert str(exc_info.value) == (
        "[auto_extend] ERROR: no H1 data for NZD_USD 2025-12-01 00:00:00+00:00 - 2026-01-20 00:00:00+00:00")
    assert fake.starts == [START] and sleeps == [] and capsys.readouterr().out == ""


def test_every_auto_extend_request_retries_including_the_post_guard_refetch(monkeypatch, capsys, sleeps):
    """Iteration 0 is 'too early'; iteration 1's request fails once; the guard (max_extend_iters=2) then trips and
    the post-loop re-fetch fails once too — each recovers."""
    monkeypatch.setattr(run_replay, "identify_start_scenario_1",
                        lambda df, **_k: SimpleNamespace(meta={"too_early": True, "raw_start_idx": 1}, start_idx=1))
    fake = _FakeHistory(_frame(), _http_504(), _frame(), _http_504(), _frame())
    monkeypatch.setattr(run_replay, "get_history", fake)
    df, eff_start, _ = _fetch(max_extend_iters=2, extend_days=4)
    moved = datetime(2025, 11, 27, tzinfo=timezone.utc)
    later = datetime(2025, 11, 23, tzinfo=timezone.utc)
    assert eff_start == later and len(df) == 3
    assert fake.starts == [START, moved, moved, later, later] and len(sleeps) == 2
    retries = [line for line in capsys.readouterr().out.splitlines() if " RETRY " in line]
    assert [line.split(" in 5 s")[0] for line in retries] == [
        "[auto_extend] RETRY 1/2 H1 fetch 2025-11-27 00:00:00+00:00-2026-01-20 00:00:00+00:00",
        "[auto_extend] RETRY 1/2 H1 fetch 2025-11-23 00:00:00+00:00-2026-01-20 00:00:00+00:00",
    ]
