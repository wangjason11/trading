"""The orchestrator's reversal map is the REALISED reversal (2026-09-27; zones-audit latent bug (d)).

`_run_downstream_pipeline` used to key `reversal_confirmed_by_sid` on the last `REVERSAL_CANDIDATE.meta["apply_idx"]`
per structure — a SCHEDULED apply. The candidate is emitted when the reversal is scheduled and survives rewinds; a
watch expiry (run BEFORE the pending apply in the per-candle step) discards it, and a reversal found at a step anchor
applies without any candidate. That map fed FibTracker's Scenario-1 checks (`reversal_confirmed_idx`), the ended
sid's fib terminals (`set_reversal_terminals`) and the prev-BOS line END, while KL / POI / the fib finalize / charts
read the realised `STATE_CHANGED(to=reversal)`. Now (user decision A) the map is the canonical
`structure_lifecycle.compute_reversal_idx_by_sid`, shifted to the new sid: `reversal_idx_by_new_sid`.
Reference window: every candidate realised at its apply — 0 cells (measured, 2026-09-27).
"""
from __future__ import annotations

import contextlib
import copy
import io

import pytest

import engine_v2.zones.fib_tracker as ft
from engine_v2.pipeline.orchestrator import _run_downstream_pipeline
from engine_v2.structure.event_fields import CTS_UPDATED_RAW_VIA
from engine_v2.structure.structure_engine import compute_bounded_structure
from engine_v2.tests._event_factory import make_bos_confirmed, make_cts_established, make_cts_updated
from engine_v2.tests.test_ms_stop_after_cts import _make_double_rewind_data
from engine_v2.tests.test_unified_probe import _prepare_df


def _ms(end_idx):
    with contextlib.redirect_stdout(io.StringIO()):
        return compute_bounded_structure(_prepare_df(_make_double_rewind_data()), 0, 1, end_idx=end_idx)


def _downstream(res, events, cap):
    with contextlib.redirect_stdout(io.StringIO()):
        return _run_downstream_pipeline(res.df, events, 1, fib_mode="h1", lifecycle_cap=cap)


def _cands(events):
    return [(e.idx, e.meta["apply_idx"]) for e in events if e.type == "REVERSAL_CANDIDATE"]


def _reversals(events):
    return [e.idx for e in events if e.type == "STATE_CHANGED" and e.meta.get("to") == "reversal"]


@pytest.mark.parametrize("end_idx", [13, 16])
def test_a_discarded_reversal_candidate_ends_no_fib(end_idx):
    """`_make_double_rewind_data` bounded at 13 / 16: candle 4 close-breaks BOS_0 and schedules a reversal (candidate
    anchor 4, apply 9 == its expiry); at 9 the expiry runs first, discards it and rewinds; cycle 1 is re-established
    at 12 and nothing reverses. The cycle-1 fib must not end in a phantom 'reversal' at 9 (before its own start):
    it ends at the run's cap, like the KL zones."""
    res = _ms(end_idx)
    assert _cands(res.events) == [(4, 9)] and _reversals(res.events) == []          # a dead candidate
    assert [(e.meta["cycle_id"], e.idx) for e in res.events if e.type == "CTS_ESTABLISHED"] == [(0, 2), (1, 12)]
    fibs = {(f.structure_id, f.cycle_id): f for f in _downstream(res, res.events, end_idx)["fib_states"]}
    f = fibs[(0, 1)]
    assert (f.start_idx, f.end_idx, f.end_reason) == (12, end_idx, "lifecycle_end")


def test_the_realised_reversal_still_ends_the_fib():
    """Positive control, the full fixture: the J2 candidate (apply 17) realises at 17 -> 'reversal' at 17."""
    res = _ms(None)
    assert _cands(res.events) == [(4, 9), (14, 17)] and _reversals(res.events) == [17]
    fibs = {(f.structure_id, f.cycle_id): f for f in _downstream(res, res.events, None)["fib_states"]}
    assert (fibs[(0, 1)].end_idx, fibs[(0, 1)].end_reason) == (17, "reversal")


def test_every_reader_gets_the_realised_reversal_not_the_last_candidate(monkeypatch):
    """A reversal can realise with NO matching candidate (found at a step anchor): the stream keeps only the
    discarded J1 candidate (apply 9) while sid 0 reverses at 17, and sid 1 (down) follows — established
    retroactively BEFORE the reversal (as on H1: sid 1's first ESTs 703 / 748 < sid 0's reversal 902), with raw
    updates at 14 (known before 17) and 17. Every reader of the map — FibTracker's Scenario-1 argument on sid 1's
    CTS_ESTABLISHED and CTS_UPDATED, the ended sid's terminals, the prev-BOS line END (the first sid-1 CTS known
    at/after the reversal: the update at 17, not the EST at 13 a 9 would pick) — gets 17, never 9."""
    res = _ms(None)
    events = [e for e in res.events if not (e.type == "REVERSAL_CANDIDATE" and e.meta["apply_idx"] == 17)]
    assert _cands(events) == [(4, 9)] and _reversals(events) == [17]
    sid1 = dict(structure_id=1, cycle_id=0, struct_direction=-1)
    events += [
        make_bos_confirmed(bos_anchor_idx=11, confirmed_at=13, price=0.60720, **sid1),
        make_cts_established(cts_anchor_idx=13, confirmed_at=13, price=0.60380, **sid1),
        make_cts_updated(cts_anchor_idx=14, via=CTS_UPDATED_RAW_VIA, price=0.58980, meta=dict(sid1)),
        make_cts_updated(cts_anchor_idx=17, via=CTS_UPDATED_RAW_VIA, price=0.58780, meta=dict(sid1)),
    ]
    seen = {"est": [], "upd": [], "terminals": []}
    real_est, real_upd, real_term = (ft.FibTracker.on_cts_established, ft.FibTracker.on_cts_updated,
                                     ft.FibTracker.set_reversal_terminals)

    def est(self, ev, df, bos_idx, bos_price, reversal_confirmed_idx=None, *a, **k):
        seen["est"].append((ev.meta["structure_id"], reversal_confirmed_idx))
        return real_est(self, ev, df, bos_idx, bos_price, reversal_confirmed_idx, *a, **k)

    def upd(self, ev, df, reversal_confirmed_idx=None):
        seen["upd"].append((ev.meta["structure_id"], reversal_confirmed_idx))
        return real_upd(self, ev, df, reversal_confirmed_idx)

    def term(self, m):
        seen["terminals"].append(dict(m))
        return real_term(self, m)

    monkeypatch.setattr(ft.FibTracker, "on_cts_established", est)
    monkeypatch.setattr(ft.FibTracker, "on_cts_updated", upd)
    monkeypatch.setattr(ft.FibTracker, "set_reversal_terminals", term)
    out = _downstream(res, copy.deepcopy(events), None)
    assert (1, 17) in seen["est"] and (1, 17) in seen["upd"]
    assert all(rv in (None, 17) for _s, rv in seen["est"] + seen["upd"])
    assert seen["terminals"] == [{1: 17}]
    assert [(ln["structure_id"], ln["end_idx"]) for ln in out["prev_bos_lines"]] == [(1, 17)]   # a 9 gives 13
