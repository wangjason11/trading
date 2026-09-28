"""The M15 fetch fails loudly (2026-09-27; zones-audit latent bug "data_bridge swallows a failed chunk").

`fetch_lower_tf_data` used to print `[data_bridge] ERROR fetching …` for a failed OANDA chunk and carry on: one
2026-09-22 audit run silently got 2500 of 4228 M15 candles and exited 0. Now a transient chunk failure is retried
`_FETCH_RETRIES` (2) times, then the fetch raises; an all-empty fetch raises too; nothing on the way up catches it.

The `/compare` §2b / `/commit-save` 4b / WORKFLOWS fetch gate reads three log families these tests pin: the
`[data_bridge] Fetched N … in K chunks` line, the `[multi_tf:dual]` prefix every M15-on run prints first, and the
ASCII prefix `[multi_tf:dual] no triggers` (its em dash reaches run.log as cp1252 byte 0x97 — never grep it).
"""
from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from engine_v2.multitf import data_bridge
from engine_v2.pipeline.orchestrator import _run_multi_tf_dual

START = pd.Timestamp("2025-11-15", tz="UTC")
END = START + timedelta(days=42)   # three 14-day chunks


def _chunk(chunk_start, n=3):
    return pd.DataFrame({
        "time": pd.date_range(chunk_start, periods=n, freq="15min", tz="UTC"),
        "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "volume": 1,
    })


class _FakeHistory:
    """`get_history` stand-in: `script[k]` is the list of outcomes for chunk k's successive requests — a DataFrame
    (returned) or an exception (raised); the last outcome repeats."""

    def __init__(self, script):
        self.script = script
        self.calls = []   # chunk index per request

    def __call__(self, *, pair, timeframe, start, end):
        k = int((start - START.to_pydatetime()) / timedelta(days=14))
        self.calls.append(k)
        outcomes = self.script[k]
        outcome = outcomes[min(self.calls.count(k) - 1, len(outcomes) - 1)]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture
def sleeps(monkeypatch):
    waited = []
    monkeypatch.setattr(data_bridge.time, "sleep", waited.append)
    return waited


def _http_504():
    return RuntimeError("OANDA candles request failed 504: gateway timeout")


def test_a_clean_fetch_returns_every_chunk_and_prints_the_gate_line(monkeypatch, capsys, sleeps):
    fake = _FakeHistory({k: [_chunk(START + timedelta(days=14 * k))] for k in range(3)})
    monkeypatch.setattr(data_bridge, "get_history", fake)
    df = data_bridge.fetch_lower_tf_data("NZD_USD", "M15", START, END)
    assert len(df) == 9 and df["time"].is_monotonic_increasing
    assert fake.calls == [0, 1, 2] and sleeps == []
    # the exact shape the fetch gate's EXPECTED_FETCH matches
    assert capsys.readouterr().out == "[data_bridge] Fetched 9 M15 candles for NZD_USD in 3 chunks\n"


def test_a_chunk_that_keeps_failing_raises_after_two_retries_and_fetches_nothing_after_it(
        monkeypatch, capsys, sleeps):
    fake = _FakeHistory({0: [_chunk(START)], 1: [_http_504()], 2: [_chunk(START)]})
    monkeypatch.setattr(data_bridge, "get_history", fake)
    with pytest.raises(RuntimeError) as exc_info:
        data_bridge.fetch_lower_tf_data("NZD_USD", "M15", START, END)
    msg = str(exc_info.value)
    assert msg.startswith("[data_bridge] ERROR fetching M15 chunk 2025-11-29 00:00:00+00:00-2025-12-13 00:00:00+00:00 "
                          "after 2 retries: OANDA candles request failed 504")
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert fake.calls == [0, 1, 1, 1]          # 1 + 2 retries on chunk 1; chunk 2 never requested
    assert sleeps == [data_bridge._FETCH_RETRY_WAIT_S] * 2 and data_bridge._FETCH_RETRY_WAIT_S > 0
    out = capsys.readouterr().out.splitlines()
    assert [line.split(" M15 chunk")[0] for line in out] == ["[data_bridge] RETRY 1/2", "[data_bridge] RETRY 2/2"]
    assert not any("Fetched" in line for line in out)


def test_a_transient_failure_recovers_into_the_complete_frame(monkeypatch, capsys, sleeps):
    fake = _FakeHistory({0: [_chunk(START)],
                         1: [_http_504(), requests.ConnectionError("reset"), _chunk(START + timedelta(days=14))],
                         2: [_chunk(START + timedelta(days=28))]})
    monkeypatch.setattr(data_bridge, "get_history", fake)
    df = data_bridge.fetch_lower_tf_data("NZD_USD", "M15", START, END)
    assert len(df) == 9
    assert fake.calls == [0, 1, 1, 1, 2]
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("[data_bridge] RETRY 1/2 M15 chunk") and out[0].endswith("504: gateway timeout")
    assert out[1].startswith("[data_bridge] RETRY 2/2 M15 chunk") and out[1].endswith(": reset")
    assert out[2] == "[data_bridge] Fetched 9 M15 candles for NZD_USD in 3 chunks"


def test_a_non_transient_error_is_raised_at_once(monkeypatch, capsys, sleeps):
    fake = _FakeHistory({0: [FileNotFoundError("Could not find 'oanda.cfg'")]})
    monkeypatch.setattr(data_bridge, "get_history", fake)
    with pytest.raises(FileNotFoundError):
        data_bridge.fetch_lower_tf_data("NZD_USD", "M15", START, END)
    assert fake.calls == [0] and sleeps == [] and capsys.readouterr().out == ""


def test_one_empty_chunk_is_left_out_of_the_count_but_an_all_empty_fetch_raises(monkeypatch, capsys, sleeps):
    empty = _chunk(START, n=0)
    monkeypatch.setattr(data_bridge, "get_history",
                        _FakeHistory({0: [_chunk(START)], 1: [empty], 2: [_chunk(START + timedelta(days=28))]}))
    assert len(data_bridge.fetch_lower_tf_data("NZD_USD", "M15", START, END)) == 6
    assert capsys.readouterr().out == "[data_bridge] Fetched 6 M15 candles for NZD_USD in 2 chunks\n"

    monkeypatch.setattr(data_bridge, "get_history", _FakeHistory({0: [empty], 1: [empty], 2: [empty]}))
    with pytest.raises(RuntimeError, match=r"^\[data_bridge\] ERROR: no M15 data for NZD_USD "):
        data_bridge.fetch_lower_tf_data("NZD_USD", "M15", START, END)


_H1 = pd.DataFrame({"time": pd.date_range("2025-11-17", periods=40, freq="h", tz="UTC")})


def _run_dual(**triggers):
    kw = dict(first_confluence_triggers=[], subsequent_confluence_triggers=[], subsequent_counter_triggers=[])
    kw.update(triggers)
    return _run_multi_tf_dual(_H1, [], [], [], [], {}, None, **kw)


def test_the_dual_driver_lets_a_fetch_failure_propagate(monkeypatch):
    """No WARNING-and-return: a failed M15 fetch ends the replay (no timing block → the gate FAILs)."""
    monkeypatch.setattr("engine_v2.multitf.uc1_trigger.detect_uc1_triggers", lambda *a, **k: [SimpleNamespace()])

    def _fail(*_a, **_k):
        raise RuntimeError("[data_bridge] ERROR fetching M15 chunk a-b after 2 retries: 504")

    monkeypatch.setattr("engine_v2.multitf.data_bridge.fetch_lower_tf_data", _fail)
    with pytest.raises(RuntimeError, match=r"^\[data_bridge\] ERROR fetching M15 chunk"):
        _run_dual()


def test_a_no_trigger_run_prints_the_gate_na_line_and_never_fetches(monkeypatch, capsys):
    """The fetch gate's N/A branch keys on these two ASCII prefixes (compare skill §2b) — reword a line here and
    the gate falls back to FAIL, so move the gate's copies (compare §2b, commit-save 4b, WORKFLOWS) with it."""
    def _no_fetch(*_a, **_k):
        raise AssertionError("a no-trigger run must not fetch M15")

    monkeypatch.setattr("engine_v2.multitf.data_bridge.fetch_lower_tf_data", _no_fetch)
    assert _run_dual() == ([], [])
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 2 and all(line.startswith("[multi_tf:dual] ") for line in out)
    assert out[1].startswith("[multi_tf:dual] no triggers")
    assert not any("[data_bridge]" in line for line in out)
